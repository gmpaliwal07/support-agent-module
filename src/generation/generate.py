"""Generate a support reply from intent and similar historical cases."""
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from ollama import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "classification"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "escalation"))

from predict import IntentClassifier  # noqa: E402
from query import HistoricalCaseRetriever  # noqa: E402
from decide import decide_escalation, EscalationDecision  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

GENERATION_MODEL = "gpt-oss:120b-cloud"
RETRIEVAL_K = 3

_URL_RE = re.compile(r"https?://\S+")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_PROMISE_RE = re.compile(
    r"\b(will refund|guaranteed?|we (will|have) (credited|refunded|processed) \$?\d)",
    re.I,
)

def _sanitize_for_prompt(text: str) -> str:
    text = _URL_RE.sub("[support link]", text)
    text = re.sub(r"@\d{4,}", "[account]", text)  # anonymized account-id mentions
    return text

SYSTEM_PROMPT = """You are a customer support agent for Uber. Draft a reply to the customer's message below.

Rules:
- Be empathetic, concise, and professional.
- If any retrieved historical case has a REAL RESOLUTION (not a generic template), use it as a model for what a substantive resolution looks like  but do not fabricate specific amounts, dates, or account details that aren't in the current customer's message.
- If retrieved cases are only generic templates, be honest: acknowledge the specific issue, and direct the customer to the appropriate next step (in-app support, DM with account details)  do not pretend to resolve something you have no authority or information to resolve.
- Never invent facts about the customer's account, ride, or order.
- Never include any URL, link, phone number, or account identifier in your reply  direct the customer to "the in-app Help Center" or "our support form" generically instead, since any specific link from historical examples belongs to a different customer's case.
- Keep the reply under 3 sentences.
"""

@dataclass
class GenerationResult:
    customer_text: str
    predicted_intent: str
    intent_confidence: float
    retrieved_cases: list[dict] = field(default_factory=list)
    has_grounded_resolution: bool = False
    drafted_reply: str = ""
    guardrail_flags: list[str] = field(default_factory=list)
    escalation_decision: str = ""
    escalation_reasons: list[str] = field(default_factory=list)


def _scan_output(reply: str) -> list[str]:
    flags = []
    if _URL_RE.search(reply):
        flags.append("contains_url")
    if _EMAIL_RE.search(reply):
        flags.append("contains_email")
    if _PHONE_RE.search(reply):
        flags.append("contains_phone")
    if _PROMISE_RE.search(reply):
        flags.append("fabricated_specific_commitment")
    if not reply.strip():
        flags.append("empty_reply")
    return flags


class ResponseGenerator:
    """Loads models once and drafts grounded replies."""
    def __init__(self) -> None:
        logger.info("loading classifier...")
        self.classifier = IntentClassifier()
        logger.info("loading retriever...")
        self.retriever = HistoricalCaseRetriever()
        self.llm_client = self._get_llm_client()

    @staticmethod
    def _get_llm_client() -> Client:
        api_key = os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            raise RuntimeError("OLLAMA_API_KEY not set in .env")
        return Client(host="https://ollama.com", headers={"Authorization": f"Bearer {api_key}"})

    def _build_prompt(self, customer_text: str, intent: str, retrieved: list[dict]) -> str:
        lines = [
            f"Customer message: {customer_text}",
            f"Predicted issue category: {intent}",
            "",
            "Similar historical cases:",
        ]
        if not retrieved:
            lines.append("(none found)")
        for i, case in enumerate(retrieved, 1):
            tag = "REAL RESOLUTION" if case["has_real_resolution"] else "template only"
            sanitized_resolution = _sanitize_for_prompt(case["resolution_text"])
            sanitized_customer = _sanitize_for_prompt(case["customer_text"])
            lines.append(
                f"{i}. [{tag}, similarity={case['similarity']:.2f}] "
                f"Customer said: \"{sanitized_customer}\" -> "
                f"Uber replied: \"{sanitized_resolution}\""
            )
        return "\n".join(lines)

    def generate(self, customer_text: str) -> GenerationResult:
        if not customer_text or not customer_text.strip():
            raise ValueError("customer_text must be non-empty")

        prediction = self.classifier.predict(customer_text)
        retrieved = self.retriever.retrieve(customer_text, k=RETRIEVAL_K)
        has_grounded_resolution = any(c["has_real_resolution"] for c in retrieved)

        prompt = self._build_prompt(customer_text, prediction["intent"], retrieved)

        try:
            response = self.llm_client.chat(
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.3},
            )
            drafted_reply = response["message"]["content"].strip()
        except Exception:
            logger.exception("LLM generation failed for text: %.80s", customer_text)
            drafted_reply = ""

        guardrail_flags = _scan_output(drafted_reply)
        if guardrail_flags:
            logger.warning("guardrail flags on generated reply: %s", guardrail_flags)

        escalation = decide_escalation(
            predicted_intent=prediction["intent"],
            intent_confidence=prediction["confidence"],
            has_grounded_resolution=has_grounded_resolution,
            guardrail_flags=guardrail_flags,
        )

        return GenerationResult(
            customer_text=customer_text,
            predicted_intent=prediction["intent"],
            intent_confidence=prediction["confidence"],
            retrieved_cases=retrieved,
            has_grounded_resolution=has_grounded_resolution,
            drafted_reply=drafted_reply,
            guardrail_flags=guardrail_flags,
            escalation_decision=escalation.decision.value,
            escalation_reasons=escalation.reasons,
        )


if __name__ == "__main__":
    generator = ResponseGenerator()

    test_messages = [
        "My driver cancelled the ride and I still got charged a fee, this is unfair",
        "I left my phone in the car, how do I get it back",
        "The driver was really rude and made me uncomfortable",
        "charges you a cancellation fee when the driver isn't even moving his car",
    ]

    for msg in test_messages:
        result = generator.generate(msg)
        print(f"\n{'='*70}")
        print(f"CUSTOMER: {result.customer_text}")
        print(f"INTENT: {result.predicted_intent} (confidence: {result.intent_confidence:.3f})")
        print(f"HAS GROUNDED RESOLUTION: {result.has_grounded_resolution}")
        print(f"DRAFTED REPLY: {result.drafted_reply}")
        print(f"DECISION: {result.escalation_decision.upper()}")
        print(f"REASONS: {result.escalation_reasons}")
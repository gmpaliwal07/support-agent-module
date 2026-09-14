import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

FORCE_ESCALATE_INTENTS = {"driver_safety_misconduct"}

CONFIDENCE_THRESHOLD = 0.85

class EscalationDecision(str, Enum):
    AUTO_HANDLE = "auto_handle"
    ESCALATE = "escalate"

@dataclass
class EscalationResult:
    decision: EscalationDecision
    reasons: list[str]

def decide_escalation(
    predicted_intent: str,
    intent_confidence: float,
    has_grounded_resolution: bool,
    guardrail_flags: list[str],
) -> EscalationResult:
    reasons: list[str] = []

    if predicted_intent in FORCE_ESCALATE_INTENTS:
        reasons.append(f"safety-critical intent ({predicted_intent}) always escalates")
        return EscalationResult(EscalationDecision.ESCALATE, reasons)

    if guardrail_flags:
        reasons.append(f"guardrail flags triggered on generated reply: {guardrail_flags}")
        return EscalationResult(EscalationDecision.ESCALATE, reasons)

    if intent_confidence < CONFIDENCE_THRESHOLD:
        reasons.append(
            f"classifier confidence {intent_confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}"
        )
        return EscalationResult(EscalationDecision.ESCALATE, reasons)

    if not has_grounded_resolution:
        reasons.append(
            "no historical case with a real (non-template) resolution was found "
            "auto-handling without precedent risks an unsubstantiated response"
        )
        return EscalationResult(EscalationDecision.ESCALATE, reasons)

    reasons.append(
        f"high confidence ({intent_confidence:.2f}), non-safety intent, "
        f"and a grounded historical resolution was found"
    )
    return EscalationResult(EscalationDecision.AUTO_HANDLE, reasons)

if __name__ == "__main__":
    test_cases = [
        {
            "predicted_intent": "driver_safety_misconduct",
            "intent_confidence": 0.99,
            "has_grounded_resolution": True,
            "guardrail_flags": [],
        },
        {
            "predicted_intent": "lost_item",
            "intent_confidence": 0.95,
            "has_grounded_resolution": False,
            "guardrail_flags": [],
        },
        {
            "predicted_intent": "ride_cancellation_dispute",
            "intent_confidence": 0.60,
            "has_grounded_resolution": True,
            "guardrail_flags": [],
        },
        {
            "predicted_intent": "payment_method_issue",
            "intent_confidence": 0.91,
            "has_grounded_resolution": True,
            "guardrail_flags": [],
        },
    ]
    for case in test_cases:
        result = decide_escalation(**case)
        print(f"\ninput: {case}")
        print(f"decision: {result.decision.value}")
        print(f"reasons: {result.reasons}")
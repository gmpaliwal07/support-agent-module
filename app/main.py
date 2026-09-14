import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "generation"))
from generate import ResponseGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("loading pipeline (classifier, retriever, LLM client)...")
    _state["generator"] = ResponseGenerator()
    logger.info("pipeline ready")
    yield
    _state.clear()

app = FastAPI(
    title="Uber Support Agent",
    description="Classifies intent, retrieves historical context, drafts a reply, and decides auto-handle vs. escalate for Uber customer support messages.",
    version="1.0.0",
    lifespan=lifespan,
)

class RespondRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Customer's message text")

class RetrievedCase(BaseModel):
    similarity: float
    customer_text: str
    resolution_text: str
    has_real_resolution: bool

class RespondResponse(BaseModel):
    predicted_intent: str
    intent_confidence: float
    has_grounded_resolution: bool
    drafted_reply: str
    guardrail_flags: list[str]
    escalation_decision: str
    escalation_reasons: list[str]
    retrieved_cases: list[RetrievedCase]

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "pipeline_loaded": "generator" in _state}

@app.post("/respond", response_model=RespondResponse)
def respond(request: RespondRequest) -> RespondResponse:
    """Run the full pipeline on a customer message: classify, retrieve,
    draft a reply, and decide whether to auto-handle or escalate.
    """
    generator: ResponseGenerator | None = _state.get("generator")
    if generator is None:
        raise HTTPException(status_code=503, detail="pipeline not yet loaded")

    try:
        result = generator.generate(request.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("generation failed")
        raise HTTPException(status_code=500, detail="internal error during generation")

    return RespondResponse(
        predicted_intent=result.predicted_intent,
        intent_confidence=result.intent_confidence,
        has_grounded_resolution=result.has_grounded_resolution,
        drafted_reply=result.drafted_reply,
        guardrail_flags=result.guardrail_flags,
        escalation_decision=result.escalation_decision,
        escalation_reasons=result.escalation_reasons,
        retrieved_cases=[
            RetrievedCase(
                similarity=c["similarity"],
                customer_text=c["customer_text"],
                resolution_text=c["resolution_text"],
                has_real_resolution=c["has_real_resolution"],
            )
            for c in result.retrieved_cases
        ],
    )
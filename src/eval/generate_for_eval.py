"""Run a small golden-set sample through the generation pipeline and save results for review."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation"))
from generate import ResponseGenerator  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
GOLDEN_PATH = EVAL_DIR / "golden_set.csv"
OUT_PATH = EVAL_DIR / "generation_eval_sample.csv"

N_SAMPLES = 40
SEED = 17

def main() -> None:
    golden = pd.read_csv(GOLDEN_PATH)
    sample = golden.sample(min(N_SAMPLES, len(golden)), random_state=SEED)
    print(f"running generation on {len(sample)} golden set examples...")

    generator = ResponseGenerator()

    rows = []
    for i, row in enumerate(sample.itertuples(), 1):
        result = generator.generate(row.clean_text) # type: ignore
        rows.append({
            "root_tweet_id": row.root_tweet_id,
            "customer_text": row.clean_text,
            "true_intent": row.intent,
            "predicted_intent": result.predicted_intent,
            "intent_confidence": result.intent_confidence,
            "has_grounded_resolution": result.has_grounded_resolution,
            "drafted_reply": result.drafted_reply,
            "guardrail_flags": ",".join(result.guardrail_flags),
            "escalation_decision": result.escalation_decision,
            "escalation_reasons": " | ".join(result.escalation_reasons),
        })
        if i % 10 == 0:
            print(f"  {i}/{len(sample)} done")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}")
    print(out["escalation_decision"].value_counts())

if __name__ == "__main__":
    main()
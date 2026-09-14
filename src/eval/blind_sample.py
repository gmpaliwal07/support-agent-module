"""Generate a blind sample for human scoring."""
from pathlib import Path
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
SCORES_PATH = EVAL_DIR / "llm_judge_scores.csv"
BLIND_OUT = EVAL_DIR / "human_scoring_blind.csv"

N_SAMPLES = 15
SEED = 3

def main() -> None:
    df = pd.read_csv(SCORES_PATH)
    sample = df.sample(min(N_SAMPLES, len(df)), random_state=SEED)

    blind_cols = ["root_tweet_id", "customer_text", "predicted_intent", "drafted_reply"]
    blind = sample[blind_cols].copy()
    for col in ["relevance", "grounding", "tone", "actionability"]:
        blind[f"human_{col}"] = ""

    blind.to_csv(BLIND_OUT, index=False)
    print(f"wrote {BLIND_OUT} ({len(blind)} rows)")

    print("\n-- Score each on 1-5: relevance, grounding, tone, actionability --")
    for i, row in blind.iterrows():
        print(f"\n[{row['root_tweet_id']}]")
        print(f"CUSTOMER: {row['customer_text']}")
        print(f"INTENT: {row['predicted_intent']}")
        print(f"REPLY: {row['drafted_reply']}")

if __name__ == "__main__":
    main()
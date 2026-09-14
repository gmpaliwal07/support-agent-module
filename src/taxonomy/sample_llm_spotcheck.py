"""Sample LLM-labeled candidates for human review across high, medium, and
low confidence levels to fairly assess labeling accuracy and human agreement.

Fill the `human_agrees` column with yes/no for each row."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CANDIDATES_PATH = DATA_DIR / "llm_labeled_candidates.csv"
OUT_PATH = DATA_DIR / "llm_spot_check.csv"

N_HIGH = 70
N_MEDIUM = 20
N_LOW = 10
SEED = 5

def main():
    df = pd.read_csv(CANDIDATES_PATH)

    high = df[df["llm_confidence"] == "high"].sample(
        min(N_HIGH, (df["llm_confidence"] == "high").sum()), random_state=SEED
    )
    medium = df[df["llm_confidence"] == "medium"].sample(
        min(N_MEDIUM, (df["llm_confidence"] == "medium").sum()), random_state=SEED
    )
    low = df[df["llm_confidence"] == "low"].sample(
        min(N_LOW, (df["llm_confidence"] == "low").sum()), random_state=SEED
    )

    sample = pd.concat([high, medium, low]).sample(frac=1, random_state=SEED)
    sample["human_agrees"] = ""  # fill in: yes / no / partial

    sample.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(sample)} rows: {len(high)} high, {len(medium)} medium, {len(low)} low)")

if __name__ == "__main__":
    main()
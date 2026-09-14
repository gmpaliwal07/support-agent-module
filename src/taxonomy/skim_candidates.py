"""Print a small random sample for each candidate intent so a human can
quickly check the hit and false-positive rates before merging."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CANDIDATES_PATH = DATA_DIR / "keyword_candidates.csv"

SAMPLE_SIZE = 15
SEED = 3

def main() -> None:
    df = pd.read_csv(CANDIDATES_PATH)
    print(f"loaded {len(df)} candidates\n")

    for intent, group in df.groupby("candidate_intent"):
        sample = group.sample(min(SAMPLE_SIZE, len(group)), random_state=SEED)
        print(f"=== {intent} (n={len(group)}) ===")
        for t in sample["clean_text"]:
            print(f"  - {t}")
        print()


if __name__ == "__main__":
    main()
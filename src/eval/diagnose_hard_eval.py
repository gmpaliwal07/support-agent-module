"""Print misclassified rows from the hard eval set for failure diagnosis."""
from pathlib import Path
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
PRED_PATH = EVAL_DIR / "hard_eval_predictions.csv"

def main():
    df = pd.read_csv(PRED_PATH)
    wrong = df[~df["correct"]]
    print(f"{len(wrong)}/{len(df)} misclassified\n")
    for _, row in wrong.iterrows():
        print(f"TEXT: {row['clean_text']}")
        print(f"  TRUE: {row['intent']}")
        print(f"  PRED: {row['predicted_intent']}")
        print()

if __name__ == "__main__":
    main()
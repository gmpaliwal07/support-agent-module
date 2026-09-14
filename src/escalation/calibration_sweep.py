import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "classification"))
from predict import IntentClassifier  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
GOLDEN_PATH = EVAL_DIR / "golden_set.csv"
HARD_EVAL_PATH = EVAL_DIR / "hard_eval_expanded_template.csv"

THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

def evaluate_set(clf: IntentClassifier, df: pd.DataFrame, name: str) -> None:
    print(f"\n=== {name} (n={len(df)}) ===")
    predictions, confidences = [], []
    for text in df["clean_text"].astype(str):
        result = clf.predict(text)
        predictions.append(result["intent"])
        confidences.append(result["confidence"])

    df = df.copy()
    df["predicted_intent"] = predictions
    df["confidence"] = confidences
    df["correct"] = df["predicted_intent"] == df["intent"]

    overall_acc = accuracy_score(df["intent"], df["predicted_intent"])
    print(f"overall accuracy (no threshold): {overall_acc:.3f}")
    print(f"\n{'threshold':<10}{'coverage':<12}{'accuracy_at_threshold':<22}{'n_passing'}")

    for t in THRESHOLDS:
        passing = df[df["confidence"] >= t]
        coverage = len(passing) / len(df)
        acc_at_t = passing["correct"].mean() if len(passing) > 0 else float("nan")
        print(f"{t:<10}{coverage:<12.2%}{acc_at_t:<22.3f}{len(passing)}")

def main() -> None:
    clf = IntentClassifier()
    golden = pd.read_csv(GOLDEN_PATH)
    hard = pd.read_csv(HARD_EVAL_PATH)

    evaluate_set(clf, golden, "Golden set (ideal)")
    evaluate_set(clf, hard, "Hard eval set")

if __name__ == "__main__":
    main()
from pathlib import Path
import pandas as pd
 
EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
PRED_PATH = EVAL_DIR / "hard_eval_predictions.csv"
 
def main():
    df = pd.read_csv(PRED_PATH)
 
    print("general_complaint predicted as something else (false negatives)")
    fn = df[(df["intent"] == "general_complaint") & (~df["correct"])]
    for _, row in fn.iterrows():
        print(f"TEXT: {row['clean_text']}")
        print(f"  PRED: {row['predicted_intent']}\n")
 
    print("\nsomething else predicted as general_complaint (false positives)")
    fp = df[(df["predicted_intent"] == "general_complaint") & (~df["correct"])]
    for _, row in fp.iterrows():
        print(f"TEXT: {row['clean_text']}")
        print(f"  TRUE: {row['intent']}\n")
 
if __name__ == "__main__":
    main()
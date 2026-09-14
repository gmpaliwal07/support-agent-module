from pathlib import Path
 
import pandas as pd
 
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
 
TRAIN_PATH = DATA_DIR / "training_set.csv"
CANDIDATES_PATH = DATA_DIR / "llm_labeled_candidates.csv"
 
def main():
    train = pd.read_csv(TRAIN_PATH)
    candidates = pd.read_csv(CANDIDATES_PATH)
 
    candidates["label_source"] = "llm_bulk"
 
    cols = ["conversation_id", "root_tweet_id", "text", "clean_text", "intent", "label_source"]
    before = len(train)
    train = pd.concat([train[cols], candidates[cols]], ignore_index=True)
    train = train.drop_duplicates(subset="root_tweet_id")
 
    print(f"training set: {before} -> {len(train)} rows (+{len(train) - before})")
    print(train["intent"].value_counts())
 
    train.to_csv(TRAIN_PATH, index=False)
    print(f"\nwrote {TRAIN_PATH}")
 
 
if __name__ == "__main__":
    main()
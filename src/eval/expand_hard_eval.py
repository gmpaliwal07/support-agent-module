from pathlib import Path
import pandas as pd
 
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
 
CLUSTERED_PATH = DATA_DIR / "uber_roots_clustered.csv"
TRAIN_PATH = DATA_DIR / "training_set.csv"
GOLDEN_PATH = EVAL_DIR / "golden_set.csv"
HARD_EVAL_PATH = EVAL_DIR / "hard_eval_expanded_template.csv" 
OUT_PATH = EVAL_DIR / "hard_eval_expanded_template.csv"
 
N_NEW_SAMPLES = 70  # + existing 30 = ~100 total
SEED = 42

def main():
    clustered = pd.read_csv(CLUSTERED_PATH)
    train = pd.read_csv(TRAIN_PATH)
    golden = pd.read_csv(GOLDEN_PATH)
    existing_hard = pd.read_csv(HARD_EVAL_PATH)
 
    used_ids = (
        set(train["root_tweet_id"])
        | set(golden["root_tweet_id"])
        | set(existing_hard["root_tweet_id"])
    )
 
    pool = clustered[
        (clustered["cluster"] == -1) & (~clustered["root_tweet_id"].isin(used_ids))
    ]
    print(f"available untouched noise pool: {len(pool)} rows")
 
    new_sample = pool.sample(N_NEW_SAMPLES, random_state=SEED).copy()
    new_sample["intent"] = ""
 
    cols = ["conversation_id", "root_tweet_id", "text", "clean_text", "intent"]
    combined = pd.concat([existing_hard[cols], new_sample[cols]], ignore_index=True)
 
    combined.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(combined)} rows total)")
    print(f"  - {len(existing_hard)} already labeled (carried over)")
    print(f"  - {len(new_sample)} new rows need labeling")

if __name__ == "__main__":
    main()
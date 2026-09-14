from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"

CLUSTERED_PATH = DATA_DIR / "uber_roots_clustered.csv"
LABELED_POOL_PATH = DATA_DIR / "labeled_pool.csv"
OUT_PATH = EVAL_DIR / "hard_eval_template.csv"

N_SAMPLES = 30
SEED = 21

def main():
    clustered = pd.read_csv(CLUSTERED_PATH)
    labeled_pool = pd.read_csv(LABELED_POOL_PATH)
    
    already_labeled_ids = set(labeled_pool["root_tweet_id"])
    true_noise = clustered[
        (clustered["cluster"] == -1)
        & (~clustered["root_tweet_id"].isin(already_labeled_ids))
    ]
    
    print(f"true unlabeled noise pool: {len(true_noise)} rows")
    
    sample = true_noise.sample(N_SAMPLES,random_state=SEED).copy()
    sample["intent"] = ""
    
    cols = ["conversation_id", "root_tweet_id", "text", "clean_text", "intent"]
    sample[cols].to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} fill in the 'intent' column by hand")
    print("\nvalid intents:")
    print(sorted(labeled_pool["intent"].unique()))


if __name__ == "__main__":
    main()
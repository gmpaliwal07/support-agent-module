import re
from pathlib import Path
 
import pandas as pd
 
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
 
CLUSTERED_PATH = DATA_DIR / "uber_roots_clustered.csv"
LABELED_POOL_PATH = DATA_DIR / "labeled_pool.csv"
HARD_EVAL_PATH = EVAL_DIR / "hard_eval_template.csv"
OUT_PATH = DATA_DIR / "account_verification_candidates.csv"
 
PATTERN = re.compile(
    r"\b(deactivat|remove (my )?profile|remove (my )?account|"
    r"no update.*(account|status)|account.*(disabled|suspended|blocked)|"
    r"reactivate)\b",
    re.I,
)

def main():
    clustered = pd.read_csv(CLUSTERED_PATH)
    labeled_pool = pd.read_csv(LABELED_POOL_PATH)
    hard_eval = pd.read_csv(HARD_EVAL_PATH)
 
    already_used_ids = set(labeled_pool["root_tweet_id"]) | set(hard_eval["root_tweet_id"])
 
    pool = clustered[~clustered["root_tweet_id"].isin(already_used_ids)]
    matches = pool[pool["clean_text"].str.contains(PATTERN, na=False)].copy()
 
    print(f"found {len(matches)} candidate matches")
    matches["candidate_intent"] = "account_phone_verification"
    cols = ["conversation_id", "root_tweet_id", "text", "clean_text", "candidate_intent"]
    matches[cols].to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} skim before merging into training set")
 
    for t in matches["clean_text"].head(15):
        print(f"  - {t}")

if __name__ == "__main__":
    main()
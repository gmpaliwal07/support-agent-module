"""Clean keyword matches based on manual review, then merge cluster labels
and confirmed keyword candidates into the final labeled pool.

Corrections:
- Remove driver_safety_misconduct false positives.
- Move cancellation-fee complaints from ride_fare_overcharge to
  ride_cancellation_dispute."""
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DRAFT_PATH = DATA_DIR / "draft_labeled.csv"
CANDIDATES_PATH = DATA_DIR / "keyword_candidates.csv"
POOL_OUT = DATA_DIR / "labeled_pool.csv"

SAFETY_FALSE_POSITIVE_RE = re.compile(r"pushed back|uberpool to uberx", re.I)
CANCELLATION_LEAK_RE = re.compile(
    r"\bcancel(l|ling|led)?\b.*\bfee\b|\bfee\b.*\bcancel|"
    r"should not be charged if i cancel",
    re.I,
)

def clean_candidates(df: pd.DataFrame) -> pd.DataFrame:
    safety_mask = (df["candidate_intent"] == "driver_safety_misconduct") & (
        df["clean_text"].str.contains(SAFETY_FALSE_POSITIVE_RE, na=False)
    )
    dropped = int(safety_mask.sum())
    df = df[~safety_mask].copy()

    leak_mask = (df["candidate_intent"] == "ride_fare_overcharge") & (
        df["clean_text"].str.contains(CANCELLATION_LEAK_RE, na=False)
    )
    reclassified = int(leak_mask.sum())
    df.loc[leak_mask, "candidate_intent"] = "ride_cancellation_dispute"

    print(f"dropped {dropped} safety false positives")
    print(f"reclassified {reclassified} fare->cancellation leaks")
    return df


def main() -> None:
    draft = pd.read_csv(DRAFT_PATH)
    draft["label_source"] = "cluster"

    candidates = pd.read_csv(CANDIDATES_PATH)
    candidates = clean_candidates(candidates)
    candidates = candidates.rename(columns={"candidate_intent": "intent"})
    candidates["label_source"] = "keyword"

    cols = ["conversation_id", "root_tweet_id", "text", "clean_text", "intent", "label_source"]
    pool = pd.concat([draft[cols], candidates[cols]], ignore_index=True)
    pool = pool.drop_duplicates(subset="root_tweet_id")

    print(f"\nfinal labeled pool: {len(pool)} rows")
    print(pool["intent"].value_counts())

    pool.to_csv(POOL_OUT, index=False)
    print(f"\nwrote {POOL_OUT}")

if __name__ == "__main__":
    main()
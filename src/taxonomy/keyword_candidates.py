"""Find candidates for intents missing from cluster labels using keyword
matching on the unmapped pool. Results require manual review before being
added to the training set; this is not an auto-labeler."""
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
UNMAPPED_PATH = DATA_DIR / "needs_manual_label.csv"
OUT_PATH = DATA_DIR / "keyword_candidates.csv"

# keyword patterns per missing intent, built from the example phrases
# already reviewed in taxonomy.md
PATTERNS = {
    "driver_safety_misconduct": re.compile(
        r"\b(harass|assault|kick(ed)? (me|us) out|unsafe|dangerous|"
        r"disrespect|rude driver|hit and run|pushed|threat)\b", re.I
    ),
    "eats_refund_request": re.compile(
        r"\b(refund|money back)\b.*\b(food|order|eats|delivery)\b|"
        r"\b(food|order|eats|delivery)\b.*\b(refund|money back)\b", re.I
    ),
    "ride_fare_overcharge": re.compile(
        r"\b(fare|charged|price)\b.*\b(higher|more|wrong|too much|double)\b|"
        r"\bsurge\b|\bovercharg", re.I
    ),
}

def main() -> None:
    df = pd.read_csv(UNMAPPED_PATH)
    df = df[df["cluster"] == -1].copy()  # keyword search only within true noise
    print(f"searching {len(df)} noise rows")

    rows = []
    for intent, pattern in PATTERNS.items():
        matches = df[df["clean_text"].str.contains(pattern, na=False)]
        print(f"{intent}: {len(matches)} candidate matches")
        matches = matches.copy()
        matches["candidate_intent"] = intent
        rows.append(matches)

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset="root_tweet_id")
    out[
        ["conversation_id", "root_tweet_id", "text", "clean_text", "candidate_intent"]
    ].to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH} ({len(out)} rows total, needs manual confirm)")

if __name__ == "__main__":
    main()
"""for loading twcs dataset."""
import csv
import sys
from pathlib import Path
from typing import Dict

csv.field_size_limit(sys.maxsize)

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "twcs.csv"

def load_rows(path: Path = RAW_PATH) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["tweet_id"]] = row
    return rows

if __name__ == "__main__":
    rows = load_rows()
    print(f"loaded {len(rows)} rows")
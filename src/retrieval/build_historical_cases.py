import re
from pathlib import Path
import pandas as pd
 
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
THREADS_PATH = DATA_DIR / "uber_threads.csv"
OUT_PATH = DATA_DIR / "historical_cases.csv"
 
BRAND_HANDLE = "Uber_Support"
 
HANDLE_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")

def clean_text(text: str) -> str:
    text = HANDLE_RE.sub("", str(text))
    text = URL_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    threads = pd.read_csv(THREADS_PATH)
    print(f"loaded {len(threads)} thread rows")
 
    cases = []
    for conv_id, group in threads.groupby("conversation_id"):
        group = group.sort_values("turn")
        first = group.iloc[0]
        if first["inbound"] != True and first["inbound"] != "True":  # noqa: E712
            continue  # skip brand-initiated threads
 
        uber_replies = group[(group["turn"] > 1) & (group["author_id"] == BRAND_HANDLE)]
        if len(uber_replies) == 0:
            continue
        reply = uber_replies.iloc[0]
 
        cases.append({
            "conversation_id": conv_id,
            "customer_text": first["text"],
            "customer_clean": clean_text(first["text"]),
            "uber_reply_text": reply["text"],
            "uber_reply_clean": clean_text(reply["text"]),
        })
 
    out = pd.DataFrame(cases)
    print(f"built {len(out)} historical cases (customer complaint -> Uber reply pairs)")
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}")
 
if __name__ == "__main__":
    main()
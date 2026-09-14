import csv
from  datetime import  datetime
from pathlib import Path
from typing import Dict, List, Set

from load import load_rows

BRAND_HANDLE = "Uber_Support"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
THREADS_OUT = OUT_DIR / "uber_threads.csv"
ROOTS_OUT = OUT_DIR / "uber_roots.csv"

TS_FORMAT = "%a %b %d %H:%M:%S %z %Y"

def parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, TS_FORMAT)

def find_root(tid: str, rows: Dict[str,dict]) ->str:
    seen: Set[str] = {tid}
    while True:
        row  = rows.get(tid)
        if row is None:
            return tid
        parent = row["in_response_to_tweet_id"]
        if not parent or parent not in rows or parent in seen:
            return tid
        
        seen.add(parent)
        tid = parent
        
    
def collect_thread(root_id: str, rows: Dict[str, dict], max_len: int = 60) -> List[dict]:
    """DFS outward from root via response_tweet_id links, then sort by time."""
    thread: List[dict] = []
    frontier = [root_id]
    visited: Set[str] = set()
    while frontier:
        tid = frontier.pop()
        if tid in visited or tid not in rows:
            continue
        visited.add(tid)
        row = rows[tid]
        thread.append(row)
        if len(thread) > max_len:
            break
        resp = row["response_tweet_id"]
        if resp:
            frontier.extend(resp.split(","))
    thread.sort(key=lambda r: parse_ts(r["created_at"]))
    return thread

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    print(f"loaded {len(rows)}")
    
    uber_tweet_ids = [tid for tid, r in rows.items() if r["author_id"] == BRAND_HANDLE]
    print(f"{BRAND_HANDLE} tweets : {len(uber_tweet_ids)}")
    
    roots: Set[str] = set()
    for tid in uber_tweet_ids:
        roots.add(find_root(tid, rows))
    print(f"unique conersation root: {len(roots)}")
    

    thread_fields = [
        "conversation_id",
        "turn",
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
    ]
    root_fields = [
        "conversation_id",
        "root_tweet_id",
        "author_id",
        "created_at",
        "text",
        "num_turns",
    ]
    
    with open(THREADS_OUT, "w", newline="", encoding="utf-8") as tf, open(
        ROOTS_OUT, "w", newline="", encoding="utf-8"
    ) as rf:
        tw = csv.DictWriter(tf, fieldnames=thread_fields)
        rw = csv.DictWriter(rf, fieldnames=root_fields)
        tw.writeheader()
        rw.writeheader()
        
        conv_id = 0
        skipped_no_customer_root = 0
        for root in roots:
            thread = collect_thread(root, rows)
            if not thread:
                continue
            conv_id+=1
            
            for i, row in enumerate(thread, start=1):
                tw.writerow(
                    {
                        "conversation_id": conv_id,
                        "turn": i,
                        "tweet_id": row["tweet_id"],
                        "author_id": row["author_id"],
                        "inbound": row["inbound"],
                        "created_at": row["created_at"],
                        "text": row["text"],
                    }
                )
            first = thread[0]
            if first["inbound"] != "True":
                skipped_no_customer_root +=1
                continue
            
            rw.writerow(
                {
                    "conversation_id": conv_id,
                    "root_tweet_id": first["tweet_id"],
                    "author_id": first["author_id"],
                    "created_at": first["created_at"],
                    "text": first["text"],
                    "num_turns": len(thread),
                }
            )
    print(f"conversations written: {conv_id}")
    print(f"conversations skipped (non-customer root): {skipped_no_customer_root}")
    print(f"wrote {THREADS_OUT}")

if __name__ == "__main__":
    main()
import re
from pathlib import Path
import pandas as pd
 
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
THREADS_PATH = DATA_DIR / "uber_threads.csv"
CASES_PATH = DATA_DIR / "historical_cases.csv"
OUT_PATH = DATA_DIR / "historical_cases_enriched.csv"
 
BRAND_HANDLE = "Uber_Support"
 
# language suggesting a real resolution happened (not just "DM us")
RESOLUTION_RE = re.compile(
    r"\b(refund(ed)?|credit(ed)?|compensat|resolved|fixed|reactivat|"
    r"waived|apolog.*refund|processed your|issued)\b",
    re.I,
)
DEFLECTION_RE = re.compile(
    r"\b(send us a note|please dm|send us a dm|contact us via|"
    r"we've followed.up|check your inbox|reach out to us)\b",
    re.I,
)
GRATITUDE_RE = re.compile(
    r"\b(thank you|thanks|appreciate|got my refund|sorted now|resolved now)\b",
    re.I,
)

def main():
    threads = pd.read_csv(THREADS_PATH)
    cases = pd.read_csv(CASES_PATH)
    print(f"loaded {len(threads)} thread rows, {len(cases)} base cases")
 
    enriched_rows = []
    deeper_found = 0
 
    for conv_id, group in threads.groupby("conversation_id"):
        group = group.sort_values("turn")
        if len(group) < 3:
            continue 
 
        uber_msgs = group[group["author_id"] == BRAND_HANDLE]
        customer_msgs = group[group["author_id"] != BRAND_HANDLE]
 
        resolution_msg = None
        for _, row in uber_msgs.iterrows():
            if RESOLUTION_RE.search(str(row["text"])) and not DEFLECTION_RE.search(str(row["text"])):
                resolution_msg = row["text"]
                break
 
        gratitude_msg = None
        for _, row in customer_msgs.iterrows():
            if GRATITUDE_RE.search(str(row["text"])):
                gratitude_msg = row["text"]
                break
 
        if resolution_msg is not None:
            deeper_found += 1
            enriched_rows.append({
                "conversation_id": conv_id,
                "resolution_text": resolution_msg,
                "customer_confirmation": gratitude_msg,
                "thread_length": len(group),
            })
 
    print(f"conversations with real resolution language found: {deeper_found}")
 
    enriched_df = pd.DataFrame(enriched_rows)
    merged = cases.merge(enriched_df, on="conversation_id", how="left")
 
    merged["best_resolution_text"] = merged["resolution_text"].fillna(merged["uber_reply_text"])
    merged["has_real_resolution"] = merged["resolution_text"].notna()
 
    print(f"\ntotal cases: {len(merged)}")
    print(f"cases with real resolution content: {merged['has_real_resolution'].sum()} "
          f"({merged['has_real_resolution'].mean()*100:.1f}%)")
 
    merged.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}")
 
    print("\n sample real resolutions found ")
    for _, row in merged[merged["has_real_resolution"]].head(10).iterrows():
        print(f"\nCUSTOMER: {row['customer_text'][:120]}")
        print(f"RESOLUTION: {row['resolution_text'][:150]}")
        if pd.notna(row["customer_confirmation"]):
            print(f"CONFIRMATION: {row['customer_confirmation'][:100]}")

if __name__ == "__main__":
    main()
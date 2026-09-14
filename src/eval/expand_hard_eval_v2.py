import json
import os
from pathlib import Path
 
import pandas as pd
from dotenv import load_dotenv
from ollama import Client
 
load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
 
CLUSTERED_PATH = DATA_DIR / "uber_roots_clustered.csv"
TRAIN_PATH = DATA_DIR / "training_set.csv"
GOLDEN_PATH = EVAL_DIR / "golden_set.csv"
EXISTING_HARD_PATH = EVAL_DIR / "hard_eval_expanded_template.csv"
OUT_PATH = EVAL_DIR / "hard_eval_expanded_v2.csv"
 
MODEL = "gpt-oss:120b-cloud"
N_NEW_SAMPLES = 100
SEED = 55
 
INTENTS = [
    "ride_cancellation_dispute", "ride_fare_overcharge", "eats_delivery_issue",
    "eats_refund_request", "payment_method_issue", "promo_ridepass_issue",
    "app_booking_technical", "account_phone_verification", "driver_document_onboarding",
    "driver_safety_misconduct", "lost_item", "rating_dispute", "tip_issue",
    "uberpool_routing_issue", "driver_vehicle_condition", "general_complaint",
]
 
SYSTEM_PROMPT = f"""You classify Uber customer support tweets into exactly one intent.
 
Valid intents: {", ".join(INTENTS)}
 
tip_issue is specifically about tipping (can't tip, tip option missing, tip charged wrong).
rating_dispute is specifically about star ratings, NOT tipping.
 
Respond with ONLY a JSON object, no other text:
{{"intent": "<one of the intents above>"}}
"""
 
 
def get_client() -> Client:
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY not set in .env")
    return Client(host="https://ollama.com", headers={"Authorization": f"Bearer {api_key}"})
 
 
def classify_one(client: Client, text: str) -> str | None:
    response = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(text)},
        ],
        format="json",
        options={"temperature": 0.0},
    )
    try:
        parsed = json.loads(response["message"]["content"])
        intent = parsed.get("intent")
        return intent if intent in INTENTS else None
    except (json.JSONDecodeError, KeyError):
        return None
 
 
def main() -> None:
    clustered = pd.read_csv(CLUSTERED_PATH)
    train = pd.read_csv(TRAIN_PATH)
    golden = pd.read_csv(GOLDEN_PATH)
    existing_hard = pd.read_csv(EXISTING_HARD_PATH)
 
    used_ids = (
        set(train["root_tweet_id"])
        | set(golden["root_tweet_id"])
        | set(existing_hard["root_tweet_id"])
    )
 
    pool = clustered[
        (clustered["cluster"] == -1) & (~clustered["root_tweet_id"].isin(used_ids))
    ]
    print(f"available untouched noise pool: {len(pool)} rows")
 
    sample = pool.sample(min(N_NEW_SAMPLES, len(pool)), random_state=SEED).copy()
    print(f"drafting labels for {len(sample)} new rows via {MODEL}...")
 
    client = get_client()
    labels = []
    for i, row in enumerate(sample.itertuples(), 1):
        intent = classify_one(client, row.clean_text) # type: ignore
        labels.append(intent)
        if i % 25 == 0:
            print(f"  {i}/{len(sample)} done")
 
    sample["intent"] = labels
    sample = sample.dropna(subset=["intent"])
    print(f"successfully labeled: {len(sample)}")
 
    cols = ["conversation_id", "root_tweet_id", "text", "clean_text", "intent"]
    combined = pd.concat([existing_hard[cols], sample[cols]], ignore_index=True)
 
    combined.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(combined)} total rows)")
    print(combined["intent"].value_counts())
 
 
if __name__ == "__main__":
    main()
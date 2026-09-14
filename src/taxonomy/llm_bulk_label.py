import json
import os
import time
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
HARD_EVAL_PATH = EVAL_DIR / "hard_eval_expanded_template.csv"

OUT_PATH = DATA_DIR / "llm_labeled_candidates.csv"

MODEL = "gpt-oss:120b-cloud"
N_SAMPLES = 7000  # total target, not additional
SEED = 99
CHECKPOINT_EVERY = 100

INTENTS = [
    "ride_cancellation_dispute",
    "ride_fare_overcharge",
    "eats_delivery_issue",
    "eats_refund_request",
    "payment_method_issue",
    "promo_ridepass_issue",
    "app_booking_technical",
    "account_phone_verification",
    "driver_document_onboarding",
    "driver_safety_misconduct",
    "lost_item",
    "rating_dispute",
    "uberpool_routing_issue",
    "driver_vehicle_condition",
    "general_complaint",
]

SYSTEM_PROMPT = f"""You classify Uber customer support tweets into exactly one intent.

Valid intents: {", ".join(INTENTS)}

Definitions:
- ride_cancellation_dispute: driver cancels or effectively forces cancellation, customer charged, disputes over fault
- ride_fare_overcharge: fare higher than quoted, surge disputes, duplicate ride charges
- eats_delivery_issue: food late/cold/wrong/missing, delivery execution problems
- eats_refund_request: explicit refund/money-back request tied to a food order
- payment_method_issue: card declined, can't add/remove payment method, unauthorized charges
- promo_ridepass_issue: promo code not applying, Ride Pass signup/renewal problems
- app_booking_technical: can't book/schedule a ride, app errors, verification code not sending during booking
- account_phone_verification: phone number issues, login/signup blocked, account deactivated/disabled
- driver_document_onboarding: driver-side document/background-check/vehicle registration issues
- driver_safety_misconduct: harassment, physical altercation, dangerous/reckless driving, discrimination
- lost_item: left/lost an item in the vehicle
- rating_dispute: mistaken star rating, disputes about rating impact
- uberpool_routing_issue: UberPool-specific routing/extra passenger complaints
- driver_vehicle_condition: vehicle cleanliness/smell/smoking complaints (not safety)
- general_complaint: vague frustration/venting with no specific actionable request; catch-all

Respond with ONLY a JSON object, no other text:
{{"intent": "<one of the intents above>", "confidence": "high|medium|low"}}
"""

def get_client() -> Client:
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY not set in .env")
    return Client(host="https://ollama.com", headers={"Authorization": f"Bearer {api_key}"})


def classify_one(client: Client, text) -> dict:
    cleaned_text = "" if text is None else str(text)
    response = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": cleaned_text},
        ],
        format="json",
        options={"temperature": 0.0},
    )
    content = response["message"]["content"]
    try:
        parsed = json.loads(content)
        intent = parsed.get("intent")
        confidence = parsed.get("confidence", "unknown")
        if intent not in INTENTS:
            return {"intent": None, "confidence": "invalid"}
        return {"intent": intent, "confidence": confidence}
    except (json.JSONDecodeError, KeyError):
        return {"intent": None, "confidence": "parse_error"}


def load_existing_results() -> pd.DataFrame:
    cols = ["conversation_id", "root_tweet_id", "text", "clean_text", "intent", "llm_confidence"]
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        print(f"resuming: {len(existing)} rows already labeled in {OUT_PATH.name}")
        return existing
    return pd.DataFrame(columns=cols)


def save_checkpoint(existing: pd.DataFrame, new_rows: list) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="root_tweet_id")
    combined.to_csv(OUT_PATH, index=False)
    return combined


def main():
    clustered = pd.read_csv(CLUSTERED_PATH)
    train = pd.read_csv(TRAIN_PATH)
    golden = pd.read_csv(GOLDEN_PATH)
    hard_eval = pd.read_csv(HARD_EVAL_PATH)
    already_done = load_existing_results()

    used_ids = (
        set(train["root_tweet_id"])
        | set(golden["root_tweet_id"])
        | set(hard_eval["root_tweet_id"])
        | set(already_done["root_tweet_id"])
    )

    pool = clustered[
        (clustered["cluster"] == -1) & (~clustered["root_tweet_id"].isin(used_ids))
    ]
    print(f"available untouched noise pool: {len(pool)} rows")

    remaining_target = max(0, N_SAMPLES - len(already_done))
    if remaining_target == 0:
        print(f"already have {len(already_done)} >= target {N_SAMPLES}, nothing to do")
        return

    sample = pool.sample(min(remaining_target, len(pool)), random_state=SEED).copy()
    print(f"labeling {len(sample)} more rows via {MODEL} (target total: {N_SAMPLES})...")

    client = get_client()
    buffer = []
    combined = already_done

    for i, row in enumerate(sample.itertuples(), 1):
        try:
            result = classify_one(client, row.clean_text)
        except Exception as e:
            result = {"intent": None, "confidence": f"error: {e}"}

        if result["intent"] is not None:
            buffer.append({
                "conversation_id": row.conversation_id,
                "root_tweet_id": row.root_tweet_id,
                "text": row.text,
                "clean_text": row.clean_text,
                "intent": result["intent"],
                "llm_confidence": result["confidence"],
            })

        if i % CHECKPOINT_EVERY == 0:
            combined = save_checkpoint(combined, buffer)
            buffer = []
            print(f"  {i}/{len(sample)} done  checkpoint saved ({len(combined)} total rows in file)")

        time.sleep(0.05)

    if buffer:
        combined = save_checkpoint(combined, buffer)

    print(f"\ndone. total rows in {OUT_PATH}: {len(combined)}")
    print(combined["intent"].value_counts())
    print("\nconfidence distribution:")
    print(combined["llm_confidence"].value_counts())

if __name__ == "__main__":
    main()
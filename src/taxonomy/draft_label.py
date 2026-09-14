from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

CLUSTERED_PATH = DATA_DIR /"uber_roots_clustered.csv"
DRAFT_OUT = DATA_DIR / "draft_labeled.csv"
UNMAPPED_OUT = DATA_DIR / "needs_manual_label.csv"


CLUSTER_TO_INTENT = {
    5: "lost_item",
    1: "promo_ridepass_issue",
    8: "app_booking_technical",
    2: "promo_ridepass_issue",
    6: "driver_document_onboarding",
    30: "payment_method_issue",
    75: "general_complaint",
    3: "rating_dispute",
    61: "account_phone_verification",
    32: "ride_cancellation_dispute",
    54: "ride_cancellation_dispute",
    12: "uberpool_routing_issue",
    0: "eats_delivery_issue",
    13: "app_booking_technical",
    45: "general_complaint",
    4: "driver_vehicle_condition",
    9: "general_complaint",
    10: "eats_delivery_issue",
    -1: None,  # noise   always unmapped
}

def main() -> None:
    df = pd.read_csv(CLUSTERED_PATH)
    print(f"loaded {len(df)} clustered rows")
 
    df["intent"] = df["cluster"].map(CLUSTER_TO_INTENT)
 
    mapped_mask = df["intent"].notna()
    draft = df[mapped_mask].copy()
    draft["label_source"] = "cluster"
 
    unmapped = df[~mapped_mask].copy()
    unmapped["label_source"] = "unmapped"
 
    print(f"draft-labeled (from named clusters): {len(draft)}")
    print(f"needs manual label (noise + unnamed clusters): {len(unmapped)}")
    print()
    print("draft label distribution:")
    print(draft["intent"].value_counts())
 
    keep_cols = [
        "conversation_id",
        "root_tweet_id",
        "text",
        "clean_text",
        "cluster",
        "intent",
        "label_source",
    ]
    draft[keep_cols].to_csv(DRAFT_OUT, index=False)
 
    unmapped_cols = [
        "conversation_id",
        "root_tweet_id",
        "text",
        "clean_text",
        "cluster",
        "label_source",
    ]
    unmapped[unmapped_cols].to_csv(UNMAPPED_OUT, index=False)
 
    print(f"\nwrote {DRAFT_OUT}")
    print(f"wrote {UNMAPPED_OUT}")
 
 
if __name__ == "__main__":
    main()
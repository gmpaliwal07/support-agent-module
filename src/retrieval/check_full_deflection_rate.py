import re
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "twcs.csv"
BRAND_HANDLE = "Uber_Support"

DEFLECTION_RE = re.compile(
    r"\b(send us a note|please dm|send us a dm|contact us via|"
    r"we've followed.up|check your inbox|reach out to us|here to help|"
    r"connect with us)\b",
    re.I,
)
RESOLUTION_RE = re.compile(
    r"\b(refund(ed)?|credit(ed)?|compensat|resolved|fixed|reactivat|"
    r"waived|processed your|issued)\b",
    re.I,
)


def main():
    df = pd.read_csv(RAW_PATH)
    uber = df[df["author_id"] == BRAND_HANDLE].copy()
    print(f"total Uber_Support tweets: {len(uber)}")

    uber["is_deflection"] = uber["text"].str.contains(DEFLECTION_RE, na=False)
    uber["has_resolution_language"] = uber["text"].str.contains(RESOLUTION_RE, na=False)
    uber["both"] = uber["is_deflection"] & uber["has_resolution_language"]

    n_deflection = uber["is_deflection"].sum()
    n_resolution = uber["has_resolution_language"].sum()
    n_both = uber["both"].sum()
    n_neither = len(uber) - uber["is_deflection"].sum() - uber["has_resolution_language"].sum() + n_both

    print(f"\ndeflection language: {n_deflection} ({n_deflection/len(uber)*100:.1f}%)")
    print(f"resolution language: {n_resolution} ({n_resolution/len(uber)*100:.1f}%)")
    print(f"both: {n_both}")
    print(f"neither (other content): {n_neither} ({n_neither/len(uber)*100:.1f}%)")

    print("\n sample of resolution-language tweets (not deflection) ")
    pure_resolution = uber[uber["has_resolution_language"] & ~uber["is_deflection"]]
    print(f"count: {len(pure_resolution)}")
    for t in pure_resolution["text"].sample(min(15, len(pure_resolution)), random_state=1):
        print(f"  - {t}")

    print("\n sample of 'neither' tweets (what else is there?) ")
    neither_sample = uber[~uber["is_deflection"] & ~uber["has_resolution_language"]]
    for t in neither_sample["text"].sample(15, random_state=1):
        print(f"  - {t}")


if __name__ == "__main__":
    main()
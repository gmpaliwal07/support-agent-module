"""Check whether the high deflection-template rate (found for Uber_Support)
is specific to choosing Uber as the brand, or a general pattern across
Twitter customer support. Compares deflection vs. resolution language
rates across several major brands in the dataset.
"""
import re
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "twcs.csv"

# a spread of brands with meaningful volume, different industries
BRANDS_TO_CHECK = [
    "Uber_Support",
    "AmazonHelp",
    "AppleSupport",
    "SpotifyCares",
    "Delta",
    "AmericanAir",
    "comcastcares",
    "British_Airways",
    "XboxSupport",
    "hulu_support",
]

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

def main() -> None:
    df = pd.read_csv(RAW_PATH)
    print(f"loaded {len(df)} total rows\n")

    print(f"{'brand':<20}{'n_tweets':<10}{'deflection%':<14}{'resolution%':<14}{'neither%'}")

    results = []
    for brand in BRANDS_TO_CHECK:
        brand_df = df[df["author_id"] == brand]
        if len(brand_df) == 0:
            print(f"{brand:<20} not found in dataset")
            continue

        n = len(brand_df)
        deflection = brand_df["text"].str.contains(DEFLECTION_RE, na=False).sum()
        resolution = brand_df["text"].str.contains(RESOLUTION_RE, na=False).sum()
        deflection_pct = deflection / n * 100
        resolution_pct = resolution / n * 100
        neither_pct = 100 - deflection_pct  # rough, not mutually exclusive with resolution

        results.append({
            "brand": brand, "n": n,
            "deflection_pct": deflection_pct, "resolution_pct": resolution_pct,
        })
        print(f"{brand:<20}{n:<10}{deflection_pct:<14.1f}{resolution_pct:<14.1f}{neither_pct:.1f}")

    results_df = pd.DataFrame(results)
    print(f"\naverage deflection % across brands: {results_df['deflection_pct'].mean():.1f}")
    print(f"average resolution % across brands: {results_df['resolution_pct'].mean():.1f}")
    print(f"\nUber's deflection % rank: "
          f"{(results_df['deflection_pct'] > results_df[results_df['brand']=='Uber_Support']['deflection_pct'].iloc[0]).sum() + 1} "
          f"out of {len(results_df)}")


if __name__ == "__main__":
    main()
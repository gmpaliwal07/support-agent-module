"""Compare human and LLM judge scores and report how closely they agree."""
from pathlib import Path
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
JUDGE_SCORES_PATH = EVAL_DIR / "llm_judge_scores.csv"
HUMAN_SCORES_PATH = EVAL_DIR / "human_scoring_filled.csv"

DIMENSIONS = ["relevance", "grounding", "tone", "actionability"]

def main() -> None:
    judge = pd.read_csv(JUDGE_SCORES_PATH)
    human = pd.read_csv(HUMAN_SCORES_PATH)

    merged = human.merge(judge, on="root_tweet_id", suffixes=("", "_judge"))
    print(f"comparing {len(merged)} rows\n")

    print(f"{'dimension':<15}{'exact_match':<14}{'within_1':<12}{'mean_abs_diff'}")
    overall_exact, overall_within1, overall_diffs = [], [], []

    for dim in DIMENSIONS:
        human_col = f"human_{dim}"
        judge_col = dim
        diffs = (merged[human_col] - merged[judge_col]).abs()

        exact = (diffs == 0).mean()
        within_1 = (diffs <= 1).mean()
        mean_diff = diffs.mean()

        overall_exact.append(exact)
        overall_within1.append(within_1)
        overall_diffs.extend(diffs.tolist())

        print(f"{dim:<15}{exact:<14.1%}{within_1:<12.1%}{mean_diff:.2f}")

    print(f"\n{'OVERALL':<15}{sum(overall_exact)/len(overall_exact):<14.1%}"
          f"{sum(overall_within1)/len(overall_within1):<12.1%}"
          f"{sum(overall_diffs)/len(overall_diffs):.2f}")

    print("\nrows with disagreement > 1 point on any dimension")
    for dim in DIMENSIONS:
        human_col = f"human_{dim}"
        judge_col = dim
        disagreements = merged[(merged[human_col] - merged[judge_col]).abs() > 1]
        for _, row in disagreements.iterrows():
            print(f"[{row['root_tweet_id']}] {dim}: human={row[human_col]}, judge={row[judge_col]}")

if __name__ == "__main__":
    main()
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
POOL_PATH = DATA_DIR / "labeled_pool.csv"
TRAIN_OUT = DATA_DIR / "training_set.csv"
 
EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
GOLDEN_OUT = EVAL_DIR / "golden_set.csv"

MIN_PER_INTENT = 8
MAX_PER_INTENT = 30
SAMPLE_FRACTION = 0.02
SEED = 11

def sample_size(n: int) -> int:
    return max(MIN_PER_INTENT, min(MAX_PER_INTENT, round(n * SAMPLE_FRACTION)))
 
def main() -> None:
    df = pd.read_csv(POOL_PATH)
    print(f"loaded {len(df)} labeled rows")
 
    golden_parts = []
    for intent, group in df.groupby("intent"):
        n = sample_size(len(group))
        n = min(n, len(group))
        golden_parts.append(group.sample(n, random_state=SEED))
 
    golden = pd.concat(golden_parts).sample(frac=1, random_state=SEED)  # shuffle
    train = df.drop(golden.index)
 
    print(f"\ngolden eval set: {len(golden)} rows")
    print(golden["intent"].value_counts())
    print(f"\ntraining pool: {len(train)} rows")
 
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    golden.to_csv(GOLDEN_OUT, index=False)
    train.to_csv(TRAIN_OUT, index=False)
 
    print(f"\nwrote {GOLDEN_OUT}")
    print(f"wrote {TRAIN_OUT}")
 
if __name__ == "__main__":
    main()
import hashlib
from pathlib import Path
 
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import normalize
 
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
EMBED_CACHE_DIR = DATA_DIR / "embed_cache"
 
TRAIN_PATH = DATA_DIR / "training_set.csv"
GOLDEN_PATH = EVAL_DIR / "golden_set.csv"
HARD_EVAL_PATH = EVAL_DIR / "hard_eval_expanded_v2.csv"
 
EMBED_MODEL = "all-MiniLM-L6-v2"
 
def embed_texts(texts: list, model_name: str = EMBED_MODEL) -> np.ndarray:
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(("\n".join(texts) + model_name).encode("utf-8")).hexdigest()
    cache_path = EMBED_CACHE_DIR / f"{key}.npy"
    if cache_path.exists():
        print(f"using cached embeddings: {cache_path.name}")
        return np.load(cache_path)
 
    model = SentenceTransformer(model_name)
    embeddings = np.asarray(model.encode(texts, show_progress_bar=True))
    np.save(cache_path, embeddings)
    return embeddings

def main():
    train = pd.read_csv(TRAIN_PATH)
    golden = pd.read_csv(GOLDEN_PATH)
    hard = pd.read_csv(HARD_EVAL_PATH)
    print(f"training pool: {len(train)} rows")
    print(f"golden eval set: {len(golden)} rows")
    print(f"hard eval set: {len(hard)} rows")
 
    X_train = normalize(embed_texts(train["clean_text"].astype(str).tolist()))
    X_golden = normalize(embed_texts(golden["clean_text"].astype(str).tolist()))
    X_hard = normalize(embed_texts(hard["clean_text"].astype(str).tolist()))
 
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, train["intent"])
 
    def score(X, y, name):
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)
        macro_f1 = f1_score(y, preds, average="macro", zero_division=0)
        print(f"\n-- {name} --")
        print(f"accuracy: {acc:.4f}")
        print(f"macro F1: {macro_f1:.4f}")
        print(classification_report(y, preds, zero_division=0))
        return preds, acc, macro_f1
 
    _, golden_acc, golden_f1 = score(X_golden, golden["intent"], "Golden set (clean/clustered)")
    hard_preds, hard_acc, hard_f1 = score(X_hard, hard["intent"], "Hard eval set (true noise, hand-labeled)")
 
    hard_out = hard.copy()
    hard_out["predicted_intent"] = hard_preds
    hard_out["correct"] = hard_out["intent"] == hard_out["predicted_intent"]
    hard_out_path = EVAL_DIR / "hard_eval_predictions.csv"
    hard_out.to_csv(hard_out_path, index=False)
 
    print("\n-- Comparison --")
    print(f"golden set:    accuracy={golden_acc:.4f}  macro_f1={golden_f1:.4f}")
    print(f"hard eval set: accuracy={hard_acc:.4f}  macro_f1={hard_f1:.4f}")
    print(f"drop: {(golden_acc - hard_acc) * 100:.1f} accuracy points")
    print(f"\nwrote {hard_out_path}")

if __name__ == "__main__":
    main()
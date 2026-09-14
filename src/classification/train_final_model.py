import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
 
import numpy as np
import pandas as pd
from joblib import dump
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EMBED_CACHE_DIR = DATA_DIR / "embed_cache"
 
TRAIN_PATH = DATA_DIR / "training_set.csv"
MODEL_PATH = MODELS_DIR / "intent_classifier.joblib"
META_PATH = MODELS_DIR / "intent_classifier_meta.json"
 
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
 
 
def embed_texts(texts: list[str], model_name: str = EMBED_MODEL_NAME) -> np.ndarray:
        
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    combined_text = "\n".join(texts) + model_name
    key = hashlib.sha256(combined_text.encode("utf-8")).hexdigest()
    cache_path = EMBED_CACHE_DIR / f"{key}.npy"
    
    if cache_path.exists():
        logger.info("using cached embeddings: %s", cache_path.name)
        return np.load(cache_path)
 
    model = SentenceTransformer(model_name)
    embeddings = np.asarray(model.encode(texts, show_progress_bar=True))
    np.save(cache_path, embeddings)
    return embeddings
 
 
def train() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"training set not found at {TRAIN_PATH}")
 
    train_df = pd.read_csv(TRAIN_PATH)
    train_df = train_df.dropna(subset=["clean_text", "intent"])
    if train_df.empty:
        raise ValueError("training set is empty after dropping missing rows")
 
    logger.info("training on %d rows across %d intents", len(train_df), train_df["intent"].nunique())
 
    X = embed_texts(train_df["clean_text"].astype(str).tolist())
    X = normalize(X)
 
    clf = LogisticRegression(
        max_iter=1000, 
        class_weight="balanced"
    )
    clf.fit(X, train_df["intent"])
 
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(clf, MODEL_PATH)
    logger.info("wrote %s", MODEL_PATH)
 
    meta = {
        "embedding_model": EMBED_MODEL_NAME,
        "intents": sorted(train_df["intent"].unique().tolist()),
        "n_training_examples": len(train_df),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("wrote %s", META_PATH)
 
 
if __name__ == "__main__":
    train()
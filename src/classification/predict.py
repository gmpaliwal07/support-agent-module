import json
import logging
from pathlib import Path

import numpy as np
from joblib import load
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
EMBED_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "embed_cache"

MODEL_PATH = MODELS_DIR / "intent_classifier.joblib"
META_PATH = MODELS_DIR / "intent_classifier_meta.json"

class IntentClassifier:
    """Wraps the persisted LogisticRegression classifier + its embedding
    model."""

    def __init__(self) -> None:
        if not MODEL_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                f"trained model not found at {MODEL_PATH}. "
                f"Run `python3 src/classification/train_final_model.py` first."
            )

        self.meta: dict = json.loads(META_PATH.read_text())
        self.clf: LogisticRegression = load(MODEL_PATH)
        self.embedding_model_name: str = self.meta["embedding_model"]
        self.intents: list[str] = self.meta["intents"]
        self._embedder = SentenceTransformer(self.embedding_model_name)
        logger.info(
            "loaded classifier: %d intents, trained on %d examples at %s",
            len(self.intents), self.meta["n_training_examples"], self.meta["trained_at"],
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = np.asarray(self._embedder.encode(texts))
        return normalize(embeddings)

    def predict(self, text: str) -> dict:
        embedding = self._embed([text])
        proba = self.clf.predict_proba(embedding)[0]
        classes = self.clf.classes_

        best_idx = int(np.argmax(proba))
        return {
            "intent": classes[best_idx],
            "confidence": float(proba[best_idx]),
            "all_probabilities": dict(zip(classes.tolist(), proba.tolist())),
        }

if __name__ == "__main__":
    clf = IntentClassifier()
    test_texts = [
        "My driver cancelled and I still got charged a fee",
        "I left my wallet in the car",
        "The driver was really aggressive and scared me",
    ]
    for t in test_texts:
        result = clf.predict(t)
        print(f"\nTEXT: {t}")
        print(f"  intent: {result['intent']} (confidence: {result['confidence']:.3f})")
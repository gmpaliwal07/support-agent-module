"""Build a smaller FAISS index from a random subsample of historical
cases, for the <15-minute quick-repro path. The full 41,836-case index
(used during actual development/evaluation) takes ~25 minutes to embed
and produces a 123MB file too large for a normal git push.

This project's design intentionally uses subsampling for the quick-demo
path  this is a deliberate shortcut for reproducibility purposes, not a
compromise of the real system (which was built and evaluated on the
full dataset  see report.md and decision_log.md).
"""
import hashlib
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
EMBED_CACHE_DIR = DATA_DIR / "embed_cache"

CASES_PATH = DATA_DIR / "historical_cases_enriched.csv"
INDEX_PATH = DATA_DIR / "faiss_index_demo.bin"
METADATA_PATH = DATA_DIR / "faiss_metadata_demo.csv"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
SUBSAMPLE_SIZE = 3000
SEED = 42

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


def main() -> None:
    cases = pd.read_csv(CASES_PATH)
    print(f"loaded {len(cases)} historical cases")

    real_resolution_cases = cases[cases["has_real_resolution"]]
    other_cases = cases[~cases["has_real_resolution"]]

    n_other_needed = SUBSAMPLE_SIZE - len(real_resolution_cases)
    sampled_other = other_cases.sample(min(n_other_needed, len(other_cases)), random_state=SEED)

    subsample = pd.concat([real_resolution_cases, sampled_other]).reset_index(drop=True)
    print(f"subsample: {len(subsample)} cases "
          f"({len(real_resolution_cases)} real-resolution, guaranteed included)")

    texts = subsample["customer_clean"].fillna("").astype(str).tolist()
    embeddings = embed_texts(texts)
    embeddings = normalize(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"built demo FAISS index: {index.ntotal} vectors, dim={dim}")

    faiss.write_index(index, str(INDEX_PATH))
    subsample.to_csv(METADATA_PATH, index=False)
    print(f"wrote {INDEX_PATH}")
    print(f"wrote {METADATA_PATH}")


if __name__ == "__main__":
    main()
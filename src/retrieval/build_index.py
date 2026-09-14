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
INDEX_PATH = DATA_DIR / "faiss_index.bin"
METADATA_PATH = DATA_DIR / "faiss_metadata.csv"
 
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

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
    cases = pd.read_csv(CASES_PATH)
    print(f"loaded {len(cases)} historical cases")
 
    texts = cases["customer_clean"].fillna("").astype(str).tolist()
    embeddings = embed_texts(texts)
    embeddings = normalize(embeddings).astype("float32")
 
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
    index.add(embeddings)
    print(f"built FAISS index: {index.ntotal} vectors, dim={dim}")
 
    faiss.write_index(index, str(INDEX_PATH))
    cases.to_csv(METADATA_PATH, index=False)
    print(f"wrote {INDEX_PATH}")
    print(f"wrote {METADATA_PATH}")

if __name__ == "__main__":
    main()
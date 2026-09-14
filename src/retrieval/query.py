"""Retrieval query interface: given a new customer complaint, find the
top-k most similar historical cases and return their actual Uber_Support
resolutions, to ground the LLM's drafted reply.
"""
import re
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
EMBED_CACHE_DIR = DATA_DIR / "embed_cache"

USE_FULL_INDEX = False

if USE_FULL_INDEX:
    INDEX_PATH = DATA_DIR / "faiss_index.bin"
    METADATA_PATH = DATA_DIR / "faiss_metadata.csv"
else:
    INDEX_PATH = DATA_DIR / "faiss_index_demo.bin"
    METADATA_PATH = DATA_DIR / "faiss_metadata_demo.csv"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"

HANDLE_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")

def clean_text(text: str) -> str:
    text = HANDLE_RE.sub("", str(text))
    text = URL_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()

class HistoricalCaseRetriever:
    def __init__(self):
        self.index = faiss.read_index(str(INDEX_PATH))
        self.metadata = pd.read_csv(METADATA_PATH)
        self.model = SentenceTransformer(EMBED_MODEL)
        print(f"loaded retriever: {self.index.ntotal} cases, dim={self.index.d}")

    def retrieve(self, query_text: str, k: int = 3, candidate_pool: int = 15, real_resolution_min_sim: float = 0.75) -> list[dict]:
        """Prefer real resolutions over template matches when they are similar enough."""
        cleaned = clean_text(query_text)
        embedding = self.model.encode([cleaned])
        embedding = normalize(embedding).astype("float32")

        scores, indices = self.index.search(embedding, candidate_pool)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            row = self.metadata.iloc[idx]
            candidates.append({
                "similarity": float(score),
                "customer_text": row["customer_text"],
                "resolution_text": row["best_resolution_text"],
                "has_real_resolution": bool(row["has_real_resolution"]),
            })

        real_matches = [c for c in candidates if c["has_real_resolution"] and c["similarity"] >= real_resolution_min_sim]
        template_matches = [c for c in candidates if not c["has_real_resolution"]]

        # real resolutions above threshold go first, then fill remaining
        # slots with the best template matches by similarity
        ranked = real_matches + sorted(template_matches, key=lambda c: -c["similarity"])
        return ranked[:k]

def main():
    retriever = HistoricalCaseRetriever()

    test_queries = [
        "My driver cancelled the ride and I still got charged a fee, this is unfair",
        "My UberEats order never arrived and I want a refund",
        "I left my phone in the car, how do I get it back",
        "The driver was really rude and made me uncomfortable",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {q}")
        results = retriever.retrieve(q, k=3)
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] similarity={r['similarity']:.3f}")
            print(f"      CUSTOMER: {r['customer_text'][:150]}")
            print(f"      UBER REPLY: {r['uber_reply_text'][:150]}")

if __name__ == "__main__":
    main()
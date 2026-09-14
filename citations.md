# Citations

AI coding assistants were used in building this project. Anything
borrowed from external sources is cited below.

## Dataset

- Customer Support on Twitter (Kaggle: `thoughtvector/customer-support-on-twitter`),
  filtered to the `Uber_Support` brand handle.

## Pretrained models

- `all-MiniLM-L6-v2` (sentence-transformers) — used for intent
  classification embeddings.
- `BAAI/bge-base-en-v1.5` — used for retrieval embeddings.
- `gpt-oss:120b` (via Ollama Cloud) — used for LLM-bulk-labeling,
  reply generation, and LLM-as-judge scoring.

None of these models were fine-tuned; all used as pretrained,
off-the-shelf components. Only the Logistic Regression classifier on
top of the embeddings was trained on this project's data.

## Libraries

pandas, numpy, scikit-learn, sentence-transformers, hdbscan, umap-learn,
faiss-cpu, matplotlib, joblib, python-dotenv, ollama (Python client) —
all used via their standard public APIs, no modified/vendored code.

## External references consulted

- **Ollama Cloud API usage** (client authentication pattern, model
  naming convention for cloud models) — referenced Ollama's official
  documentation to get the correct integration pattern (`Client(host=..., headers=...)`, `model:120b-cloud` naming).
- **FAISS cosine similarity pattern** (`IndexFlatIP` + L2-normalized
  vectors as a substitute for cosine similarity search) — this is a
  standard, widely-documented FAISS technique, not from one specific
  source; used as-is, no modification.
- **HDBSCAN parameter semantics** (`min_cluster_size`, `min_samples`,
  `cluster_selection_epsilon`, `cluster_selection_method` — `eom` vs
  `leaf`) — understanding of these came from HDBSCAN's own
  documentation; parameter values themselves were determined empirically
  via the grid sweep in this project (`src/taxonomy/cluster_explore.py`),
  not copied from an external example.
- **DBCV / HDBSCAN validity index** (`hdbscan.validity.validity_index`)
  — standard library function, used as-is for cluster quality scoring.

## Everything else

All taxonomy definitions, labeling decisions, clustering parameter
choices, escalation logic design, guardrail rules, and evaluation
methodology are original work for this project, developed iteratively
and documented with reasoning in `decision_log.md`.


# Uber Support Agent

AI customer support agent for Uber_Support (Customer Support on Twitter
dataset): classifies customer intent, retrieves similar historical
cases, drafts a grounded reply, and decides whether to auto-handle or
escalate to a human with a stated reason for every decision.

Full reasoning for every design decision: [`decision_log.md`](decision_log.md).
Report (results, failure analysis, next steps): [`report/report.md`](report/report.md).
Failure analysis detail: [`report/failure_analysis.md`](report/failure_analysis.md).
Intent taxonomy: [`taxonomy.md`](taxonomy.md).

---

## Quick reproduce (headline results, under 15 minutes)

This path uses artifacts already committed to the repo (cached
embeddings, trained classifier, FAISS index) so you don't need to
re-download models or re-run multi-hour steps. It reproduces the
headline metrics reported in `report/report.md`.

### 1. Setup (~3 min)

```bash
python3 -m venv .venv
source .venv/bin/activate      # or .venv/bin/activate.fish
pip install -r requirements.txt --break-system-packages
```

Create `.env` in the project root with:

```
OLLAMA_API_KEY=your_key_here
```

(Free key at https://ollama.com only needed for the generation/
escalation demo in step 4; steps 2-3 work without it.)

### 2. Baselines + final classifier (~3 min, uses cached embeddings)

```bash
python3 src/classification/baselines.py
python3 src/eval/score_hard_eval.py
```

Reproduces: majority-class baseline, TF-IDF+LR baseline, embeddings+LR
final accuracy on both the golden set (207 examples) and hard eval set
(200 examples) the core numbers in report.md Section 4.

### 3. Escalation threshold calibration (~2 min)

```bash
python3 src/escalation/calibration_sweep.py
```

Reproduces the confidence-threshold validation in report.md Section 4/6.

### 4. Full pipeline demo: classify -> retrieve -> generate -> escalate (~2 min, needs OLLAMA_API_KEY)

```bash
python3 src/generation/generate.py
```

Runs the complete agent end-to-end on sample messages, printing intent,
confidence, retrieved historical context, drafted reply, and the
auto-handle/escalate decision with reasons.

**Note on the retrieval index:** this uses a pre-built 3,000-case demo
FAISS index (`data/processed/faiss_index_demo.bin`, ~9MB, committed to
the repo), not the full 41,836-case index used for the actual reported
results (123MB exceeds GitHub's file size limit, not committed). Per
this project's design rule (a subsample is sufficient and expected for
reproducibility purposes), this is an intentional shortcut. The demo
index includes all 106 cases with real (non-template) historical
resolutions, so it still demonstrates both auto-handle and escalate
decision paths correctly. To rebuild the full index instead:
`python3 src/retrieval/build_index.py`, then set `USE_FULL_INDEX = True`
in `src/retrieval/query.py`.

### 5. (Optional) LLM-judge + human agreement reproduction (~5 min, needs OLLAMA_API_KEY)

```bash
python3 src/eval/generate_for_eval.py   # generates 40 golden-set replies
python3 src/eval/llm_judge.py           # scores them
python3 src/eval/judge_agreement.py     # compares to the committed human scores
```

Reproduces the reply-quality and human-agreement numbers in report.md
Section 8.

### 6. (Optional) Retrieval evaluation (~1 min)

```bash
python3 src/eval/retrieval_eval.py
```

Reproduces the precision@3 / approximate recall@3 numbers in report.md
Section 8 addendum.

---

## Full pipeline from raw data

The steps above use pre-built artifacts. Rebuilding everything from the
raw Kaggle dataset is **not required to verify headline results** and
takes several hours (embedding 41k+ texts, clustering parameter sweeps,
7,000 LLM-bulk-labeling API calls). Documented here for completeness /
audit purposes only.

1. Download `twcs.csv` (Kaggle: `thoughtvector/customer-support-on-twitter`)
   into `data/raw/`.
2. `python3 src/data/reconstruct.py` reconstruct conversation threads (~1 min).
3. `python3 src/taxonomy/cluster_explore.py` embeddings + HDBSCAN clustering
   for taxonomy discovery (~10 min; set `SWEEP=True` first to reproduce the
   parameter sweep, ~5 min more).
4. `python3 src/taxonomy/draft_label.py` auto-label from named clusters.
5. `python3 src/taxonomy/keyword_candidates.py` + `finalize_labels.py`    recover intents missing from cluster mapping.
6. `python3 src/taxonomy/split_golden.py` stratified golden set split.
7. `python3 src/taxonomy/llm_bulk_label.py` LLM-assisted bulk labeling
   of the noise pool (**~1.5-2 hours** for 7,000 examples; checkpointed
   every 100 rows, safe to interrupt/resume). Requires `OLLAMA_API_KEY`.
8. `python3 src/taxonomy/merge_llm_labels.py` merge into training set.
9. `python3 src/classification/train_final_model.py` train + persist
   the final classifier.
10. `python3 src/retrieval/build_historical_cases.py` build the
    retrieval knowledge base.
11. `python3 src/retrieval/find_real_resolutions.py` enrich with real
    resolution detection.
12. `python3 src/retrieval/build_index.py` build the FAISS index
    (~25 min, downloads and runs `bge-base-en-v1.5`).
13. `python3 src/eval/expand_hard_eval_v2.py` expand the hard eval set
    from 100 to 200 rows (LLM-drafted, spot-checked decision log #22).

## Project structure

```
uber-support-agent/
├── app/                     FastAPI service (/respond endpoint)
├── data/processed/          intermediate + final labeled datasets
├── models/                  persisted trained classifier
├── src/
│   ├── data/                 thread reconstruction
│   ├── taxonomy/              clustering, labeling
│   ├── classification/        baselines, final classifier
│   ├── retrieval/             FAISS knowledge base + query
│   ├── generation/            LLM reply drafting + guardrails
│   ├── escalation/            auto-handle vs. escalate logic
│   └── eval/                  golden/hard eval sets, LLM-judge,
│                               agreement, retrieval eval
├── eval/                    golden set, hard eval set, judge scores
├── report/                  report.md, failure_analysis.md
├── decision_log.md
├── taxonomy.md
└── citations.md
```

## Notes

- Zero-cost stack throughout: open-source models (sentence-transformers,
  scikit-learn) + free-tier Ollama Cloud for LLM labeling/generation/
  judging. No paid APIs, no GPU training.
- Two embedding models are used deliberately (classification: MiniLM;
  retrieval: bge-base-en-v1.5) see decision log #13 for why.
- Taxonomy has 16 intents, not 15 `tip_issue` was split out of
  `rating_dispute` during a manual audit (decision log #18).
- All non-obvious decisions, including several corrected mistakes, are
  logged with reasoning in `decision_log.md`.

  [Full Report](report/report.md) | [Decision Log](decision_log.md) | [Failure Analysis](report/failure_analysis.md)

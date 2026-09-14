# Uber Support Agent Report

## 1. Problem framing

**Brand:** Uber_Support (56,270 outbound tweets third-largest brand
in the Customer Support on Twitter dataset).

**What "good" means here:** Uber_Support answers for two product lines
(Rides, Eats) plus cross-cutting account/payment/app issues. A good
agent must (1) route messages to a taxonomy grounded in real customer
language, not invented categories, (2) draft replies that are honest
about what it can and can't confirm never fabricating a refund,
account action, or commitment it has no authority to make, and (3)
know when it lacks evidence to resolve something, and say so.

**What we chose not to build:**

- No multi-brand generalization specialized to Uber by design.
- No fine-tuned LLM classification uses embeddings + Logistic
  Regression; generation uses a pretrained model via Ollama with
  retrieval-augmented prompting. Matches the zero-cost constraint.
- No unified embedding model MiniLM for classification (iteratively
  validated), bge-base-en-v1.5 for retrieval (purpose-built). Deliberate
  tradeoff, not an oversight (decision log #13).
- No formal probability calibration `predict_proba` used directly;
  validated empirically that it correlates with accuracy (Section 4),
  but Platt scaling wasn't applied (see #10).

## 2. Architecture

```
Customer message
      |
Intent classification (embeddings + Logistic Regression, 16 intents)
      |
Historical case retrieval (FAISS over 41,836 real Uber_Support cases)
      |
Reply generation (LLM, prompted with retrieved cases, guardrail-sanitized)
      |
Escalation decision (safety intent / guardrail flags / confidence /
                      grounded-resolution check -> auto-handle or escalate)
```

Full reasoning for every non-obvious choice: `decision_log.md` (22 entries).

## 3. Intent taxonomy

15 intents initially, later expanded to 16 (`tip_issue`, decision log
#18), derived from HDBSCAN clustering over embeddings of 41,660
customer complaints (two clustering passes see `taxonomy.md` and
decision log #6-9). Validated, not assumed: two independent samples of
unclustered "noise" points were manually reviewed and confirmed to fit
existing categories, not a missing intent.

## 4. Results vs. baselines

207-example golden evaluation set (stratified, zero training overlap):

| Model                                             | Accuracy | Macro F1 |
| ------------------------------------------------- | -------- | -------- |
| Majority class (trivial baseline)                 | 8.2%     | 1.0%     |
| TF-IDF + Logistic Regression (simple baseline)    | 87.9%    | 87.5%    |
| Embeddings (MiniLM) + Logistic Regression (final) | 88.4%    | 89.5%    |

Hard-eval-set accuracy for the final classifier: 62.5% (macro F1
48.9%) see Section 5 for why this gap matters more than the
golden-set number alone.

Escalation confidence threshold (0.85) was empirically swept, not
guessed (decision log #14): accuracy among predictions clearing it is
99.2% (golden) / 88.2% (100-row hard eval), versus 88.4%/59.0%
unfiltered confidence is a real, usable signal. (Sweep predates the
hard eval set's expansion to 200 rows decision log #22 not rerun
against the larger set.)

## 5. What is misleading about my headline number

**The 88.4% golden-set accuracy is misleading**, for three reasons:

**First**, the golden set was labeled via clustering/keyword search,
biasing it toward cleanly-phrased text. A separate hard eval set (hand-
labeled from messages that never fit any cluster) scored 43.0%
initially a 49-point gap.

**Second**, we closed part of that gap honestly, not by tuning against
the test set training-data narrowness (Section 6) was the driver.
After fixes, hard-eval accuracy reached 62.5% (on an expanded, more
reliable 200-row set decision log #22), still a **25.9-point gap**
from the golden number, which we treat as the more honest figure.

**Third**, `has_grounded_resolution` shows `True` for only ~0.3% of
real queries only 106 of 41,836 historical replies contain a real
resolution rather than a DM-deflection (verified against the full
56,270-tweet corpus, not just the subsample). **This is Uber-specific,
not general**: checking 9 other brands (Amazon, Apple, Spotify, Delta,
American Airlines, Comcast, British Airways, Xbox, Hulu) shows Uber's
64.7% deflection rate is the highest of all 10, and its 0.3% resolution
rate is the lowest. A different brand would likely show a materially
higher natural auto-handle rate this is a genuine finding about
Uber's support behavior, not a methodology artifact.

## 6. Improvement process (classifier)

| Stage                                    | Hard-eval accuracy | Gap vs. golden |
| ---------------------------------------- | ------------------ | -------------- |
| Initial (n=30)                           | 46.7%              | 45.4 pts       |
| + targeted fix for one confusion pattern | 53.3%              | 38.9 pts       |
| Expanded to n=100 (reliable)             | 43.0%              | 49.3 pts       |
| + 3,000 LLM-bulk-labeled examples        | 58.0%              | 32.3 pts       |
| + 7,000 total LLM-bulk-labeled examples  | 59.0%              | 29.4 pts       |
| + expanded hard eval to 200 rows         | 62.5%              | 25.9 pts       |

Diminishing returns confirmed empirically (last 4,000 examples added
only 1 point) stopped rather than chasing a target number (decision
log #11, test-set-overfitting risk).

**Post-hoc refinement:** a manual audit found `rating_dispute` mixed
star-rating and tipping complaints split into `tip_issue` (decision
log #18). Fixing training data alone first caused a train/eval label
mismatch that hurt golden macro F1 by 8.9 points eval sets need the
same taxonomy updates as training data. After syncing both, golden
macro F1 improved to 89.5% (from 88.8%); hard-eval was unaffected (no
tip examples in that 100-row sample at the time).

## 7. Failure analysis

Full detail in `failure_analysis.md`. Summary:

1. **Cross-category confusion** (account verification vs. driver docs)
   resolved for this pair; likely elsewhere, not fully audited.
2. **`general_complaint` structurally low-precision** inherent
   ambiguity, partially mitigated, not resolved.
3. **99.7% of historical "resolutions" are deflection templates**    dataset limitation, designed around via escalation logic.
4. **LLM-judge more lenient than a human on unverified claims** known
   LLM-judge weakness; deterministic guardrails are authoritative for
   safety-critical decisions instead.
5. **Data leakage** (retrieved context copied into a different
   customer's reply) real bug, found and fixed, verified resolved.

## 8. Evaluation harness

- **Classifier:** golden set (207) + hard eval set (200, expanded from
  100 decision log #22), full precision/recall/F1 per intent.
- **LLM-as-judge for reply quality:** 40 replies scored on 4 dimensions
  (1-5 scale). Means: relevance 4.33, grounding 5.00, tone 4.97,
  actionability 4.55.
- **Human agreement with judge:** blind single-rater comparison on 15
  replies 88.3% within-1-point, 55.0% exact match. Disagreement
  concentrated on grounding (failure mode 4). **Limitation:** single
  rater, not multiple independent annotators with Cohen's kappa a
  known gap, not overstated as rigorous.
- **Escalation outcome (40-sample run):** 100% escalated 60% no
  grounded resolution, 37.5% low confidence, 2.5% forced safety. Zero
  guardrail triggers.
- **Retrieval:** precision@3 / approx. recall@3 across 7 queries, 5+
  intents. Mean precision@3 = 0.95 (28/30 relevant); one query scored
  0.67, a genuine precision miss kept in the average, not excluded.
  True recall@k is infeasible without judging the full corpus;
  approximated as (relevant in top-3)/(relevant in top-10) mean
  approx. recall@3 = 0.35, consistent with `RETRIEVAL_K=3` favoring
  precision for LLM-grounding.

## 9. On the 100% escalation rate

This could look like the agent "doesn't work" it's instead a direct
consequence of two measured facts: the 0.3% real-resolution base rate
(Section 5) and a validated confidence threshold correctly rejecting
uncertain predictions (Section 4). The cross-brand check confirms this
reflects Uber's specific support behavior, not a general limitation a system built the same way for Apple or Hulu would likely auto-handle
more. Even when escalating, the agent still classifies, retrieves
context, and drafts a starting reply reducing human read/write time,
though this isn't empirically measured. This assistive-not-autonomous
pattern mirrors how real support-AI products (e.g. Hiver) are
positioned, rather than overselling autonomy the data can't support.

## 10. What I'd do with one more week

1. **Multi-annotator agreement** with Cohen's kappa, replacing the
   current single-rater comparison.
2. ~~Retrieval evaluation~~ **done** (Section 8). Remaining gap: 7
   queries is a small test set, not a large purpose-built one.
3. **Systematic audit of remaining intents** for the training-narrowness
   failure mode **partially done** (`tip_issue` found, decision log
   #18); ~13 intents unaudited.
4. **Unify the two embedding models** if a controlled test shows no
   regression.
5. ~~Calibrate confidence~~ **partially done**: empirical checks
   confirm the model is underconfident (safe direction, decision log
   #20). Formal Platt scaling not applied; per-intent curves were too
   noisy to trust at current sample sizes.
6. **Automated test suite** zero unit/integration tests currently.
7. **Latency/cost benchmarking** pipeline takes ~20s/message
   (dominated by model loading), not optimized or measured for scale.
8. **Prompt-injection threat modeling** guardrails cover output
   leakage/fabrication, not adversarial input designed to manipulate
   the LLM.

# Failure Analysis

Five failure modes found during development, with real examples,
hypotheses, and current status. All examples are from actual pipeline
output, not constructed.

---

## 1. Cross-category confusion: account_phone_verification vs driver_document_onboarding

Both intents share "account" vocabulary, but one is rider-side access
and the other is driver-side onboarding. The classifier consistently
confused them.

**Examples:** "how do I remove a profile," "my app says 'deactivated'"
— all true `account_phone_verification`, predicted `driver_document_onboarding`.

**Hypothesis:** Training data for `account_phone_verification` came
from one narrow cluster about phone-number/verification wording only.
Real phrasing (account disabled, profile removal) was missing, so the
model defaulted to the more populous, superficially similar class.

**Fix:** Keyword search recovered 353 more real-world examples. After
retraining: confusion resolved (0.00→0.80 F1 on the affected subset),
hard-eval accuracy 46.7%→53.3%, golden accuracy unchanged — confirming
the fix generalized rather than overfit.

**Status:** Resolved for this pair. Same risk likely applies to other
narrow-cluster intents — not systematically audited across all 16 due
to time (2 of 16 checked so far, see decision log #18).

---

## 2. `general_complaint` is structurally low-precision

Defined as a catch-all for vague frustration with no specific request.
Consistently the weakest class (24-45% recall) across every round.

**Examples:** "can you pick me up? Thnx" → predicted `app_booking_technical`;
"y'all having stupid drivers" → predicted `driver_safety_misconduct`.

**Hypothesis:** Errors scatter across many different predicted classes,
not one confusable pair — the signature of genuine ambiguity, not a
data gap. Vague text shares surface words with specific intents without
enough signal to commit to one.

**Fix attempted:** More training data raised recall 24%→45%, still
below other classes — consistent with the ambiguity hypothesis.

**Status:** Partially mitigated, not resolved. Accepted as an inherent
limitation, disclosed rather than hidden.

---

## 3. Historical "resolutions" are 99.7% deflection templates

The retrieval knowledge base was meant to ground replies in real past
resolutions. Only 106 of 41,836 cases (0.3%) contain a substantive
resolution; the rest are generic "please DM us" deflections, since real
problem-solving happened in private DMs never captured in the dataset.

**Example:** Query about a cancellation-fee dispute retrieves 3 highly
similar cases (0.92-0.93 similarity), but all 3 historical replies are
just "send us a note" — no real resolution to ground on.

**Confirmed dataset-wide, not brand-specific bias:** across the full
56,270-tweet Uber corpus, 64.7% is deflection language, 0.3% resolution.
A cross-brand check (9 other brands) found Uber's deflection rate is
the highest of all 10 checked — a genuine property of Uber's specific
public support behavior, not an artifact of the methodology.

**Fix:** Retrieval composition (real resolution vs. template) is used
as an escalation signal, not just reply content. The generation prompt
instructs the LLM to acknowledge the issue and direct to the right
channel, matching Uber's real historical behavior, rather than
fabricating a resolution it has no evidence for.

**Status:** Accepted as a dataset limitation, designed around. Main
driver of the escalation rate (see report.md Section 9).

---

## 4. LLM-as-judge is more lenient than a human on unverified claims

Comparing blind human scores to LLM-judge scores on 15 replies:
88.3% agreement within 1 point overall, but disagreements clustered on
"grounding" — always with the judge scoring higher.

**Example:** A reply stated "drivers can't see whether a promo is
applied" with no source. Human scored grounding=3 (unverified claim);
judge scored grounding=5 (fully grounded).

**Hypothesis:** The judge evaluates plausibility and internal
consistency, not whether a claim is actually sourced from the prompt's
evidence. Confident, well-phrased claims read as "grounded" even
without support.

**Fix:** None applied to the judge. The deterministic guardrail scan
(URL/email/phone/fabricated-commitment check) is treated as the
authoritative fabrication check for escalation, not the judge's
grounding score.

**Status:** Documented limitation, specific to grounding — the other 3
rubric dimensions (relevance, tone, actionability) agreed with the
human 100%/93.3%/80% within 1 point.

---

## 5. Data leakage: retrieved context copied into a different customer's reply

An early test showed the LLM copying a real support URL from a
retrieved historical case — belonging to a different customer — into
the current customer's drafted reply.

**Example:** Query about a lost phone retrieved a similar case
(0.988 similarity) whose historical reply had a real t.co link. The
first draft included that exact link, specific to someone else's case.

**Hypothesis:** Nothing discouraged the LLM from treating retrieved
example text as copyable rather than reference-only. A general RAG
risk — retrieved documents can carry sensitive content unrelated to
their grounding value.

**Fix:** Two layers. (1) URLs and account-ID mentions are stripped from
retrieved context before it reaches the prompt. (2) The generated reply
is scanned post-hoc for URLs/emails/phones/fabricated commitments; any
match forces escalation regardless of confidence.

**Status:** Resolved and verified — retested the same query, confirmed
no leak.

---

## Cross-cutting observation

Three of the five failure modes (2, 3, 4) are properties of the data or
the evaluation tooling, not bugs in the pipeline code. This shaped the
overall design: rather than forcing the classifier or generator to
compensate for structural limits, the system detects them (confidence
thresholds, retrieval composition, guardrail scans) and escalates
honestly instead of guessing confidently.

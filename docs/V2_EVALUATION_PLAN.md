# V2 Evaluation Pipeline Plan (without logprobs)

**Status:** brainstorming / pre-implementation. For discussion with research advisor.
**Goal of V2:** replace the unreliable self-reported 1–5 confidence score with a
*measured, continuous, and calibration-tested* confidence signal — without
depending on token logprobs, which our current stack cannot expose (see §1).

---

## 0. Why this plan exists (what was wrong with V1)

V1 answered "do the input features drive the verdict, or memorized facts?" by
comparing agreement rates across three arms (Baseline / A / B). The weak link was
**`conf_classification` (1–5)** — a *post-hoc, self-reported* score. Three problems:

1. **Self-report ≠ internal state.** The model narrates a confidence *after*
   deciding; it is not a measurement of the decision itself.
2. **Coarse & clustered.** A 1–5 integer bins everything and tends to pile up at
   4–5, so it can't produce a continuous confidence distribution.
3. **Never validated.** We never tested whether the score is *calibrated* (i.e.
   whether "5" actually means "more likely correct" than "3").

V2 fixes all three, and adds the analysis V1 couldn't do: **calibration**.

---

## 1. Why not logprobs (the constraint that shapes this plan)

Token logprobs would be the textbook continuous confidence signal, but our exact
stack is the one configuration where OpenAI does not return them. Three stacked
reasons:

1. **Strict structured outputs suppress logprobs on the GPT-5 family.** Our
   requests use `json_schema` + `strict: true`, which triggers *constrained
   decoding*. On GPT-5 models that path currently returns an empty `logprobs`
   array even when requested. (The same request on a GPT-4.1 model returns them —
   so it is an implementation gap, not a hard law.)
2. **Reasoning models don't expose logprobs on hidden reasoning**, and the GPT-5
   reasoning line suppresses them on the visible answer too unless reasoning is
   turned off — which still doesn't beat reason #1.
3. The transport (`top_logprobs` + `include: ["message.output_text.logprobs"]`)
   works fine on Responses + Batch; the model+format combination is the blocker.

**Decision for V2:** do not build on logprobs. Use **self-consistency sampling**
as the primary confidence signal instead. Crucially, this lets us **keep the V1
instrument unchanged** (`gpt-5.4-nano` + strict schema), preserving V1↔V2
comparability. (Logprobs remain a possible V3 path if OpenAI patches the gap or
we adopt a non-reasoning model — out of scope here.)

---

## 2. Design principles

- **Optimize for reliability over speed/cost.** Cost is not a binding constraint.
- **Keep one varied factor.** The arm manipulation (which input fields are shown)
  stays the *only* experimental variable. Every V2 change applies identically to
  all three arms.
- **Keep V1's verbalized score as a baseline to beat** — don't delete it. The
  headline V2 result is *self-consistency confidence vs verbalized confidence*,
  measured on the same ground-truth rows.
- **Measure, don't assume.** Every confidence claim is validated with calibration
  metrics against ground truth.

---

## 3. Primary confidence signal: self-consistency sampling

### 3.1 The idea (plain version)
Ask the model the *same* question N times with sampling on, and watch how much it
agrees with itself. If 9 of 10 runs say "AI-native," confidence = 0.9. If it's a
5/5 split, confidence ≈ 0.5 (max uncertainty). This is the model "voting against
itself" — disagreement across runs is a direct, continuous uncertainty signal,
and it needs no logprobs.

### 3.2 Why it's a strong choice here
- **Model-agnostic** — survives the logprob blocker entirely.
- **Keeps the V1 instrument** — no model or output-format change required.
- **Double win:** the majority vote is *more accurate* than a single pass
  (the original self-consistency benefit), *and* the vote spread gives confidence.
- **Continuous** — produces a [0,1] score and a full empirical distribution.

### 3.3 How confidence is computed (per axis)
For each company × arm, run N independent samples. For a given output axis
(e.g. `ai_native`, `subclass`):
- **Final label** = majority vote (modal class across the N runs).
- **Agreement confidence** = (count of modal class) / N.
- **Vote entropy** (richer signal) = normalized Shannon entropy over the class
  vote distribution; low entropy = high confidence. Report both.

`ai_native` (binary) is the cleanest axis for calibration; `subclass`,
`rad_score`, `cohort` get the same treatment for secondary analysis.

### 3.4 Parameters to settle (in the pilot, §7)
- **N** (samples per item): start at 10; sensitivity-check 5 vs 10 vs 20.
- **Sampling diversity source.** *Critical open question:* reasoning models may
  ignore or fix `temperature`. If so, repeated runs could be near-identical and
  self-consistency collapses. The pilot must confirm we get genuine run-to-run
  variation (via temperature and/or seed variation). If diversity is
  insufficient, fall back to a model that honors temperature for the
  confidence-study subset only.

---

## 4. Refining the golden (ground-truth) set

The calibration analysis is only as trustworthy as the gold labels. V1 had a
~22k-row Tavily evidence-grounded subset; V2 hardens it.

1. **Multi-source corroboration.** Require ≥2 independent sources to agree before
   a row earns a gold label; otherwise mark it `insufficient-evidence`.
2. **Add an explicit "unknowable" class.** Don't force a gold label where evidence
   genuinely doesn't exist — forcing one injects noise into calibration.
3. **Validate the validator.** Hand-adjudicate a few-hundred-row sample and
   measure agreement (Cohen's κ) between human labels and the Tavily-derived
   labels. This tells us how much to trust the automated gold set itself.
4. **Stratified coverage.** Ensure the gold set spans all fame quartiles (Q1–Q4),
   all subclasses (including rare ones like 1F/1G), and both cohorts — so
   calibration can be measured *per stratum*, not just in aggregate.
5. **Freeze & version it.** Snapshot the final gold set so every metric is
   reproducible against a fixed reference.

---

## 5. The evaluation core: calibration & reliability metrics

Run all of the following on the gold subset, **once for self-consistency
confidence and once for verbalized 1–5 confidence**, then compare.

| Metric | Question it answers |
|---|---|
| **Reliability diagram** | Does predicted confidence match actual accuracy? (the picture) |
| **ECE** (Expected Calibration Error) | One number: how far off the diagonal? |
| **Brier score** | Are predictions both confident *and* correct? (proper score) |
| **AUROC for error detection** | Can confidence *rank* its own mistakes? |
| **Risk–coverage curve / AURC** | "If we only trust the top X% most-confident, what's the accuracy?" |

**Headline deliverable:** a side-by-side showing self-consistency confidence is
better-calibrated than the verbalized score (lower ECE/Brier, higher error-detection
AUROC). That empirically *demonstrates* the thesis "self-reported confidence is
unreliable; measured signals are better," rather than asserting it.

---

## 6. Tiered execution design (keeps it tractable at full reliability)

- **Tier 1 — full population (269k × 3 arms), single pass.** Reuse for the
  large-scale agreement-rate / directness analysis (as in V1).
- **Tier 2 — gold subset (~22k × 3 arms), N-sample self-consistency.** This is
  where ground truth exists, so it's where calibration is measured and where the
  sampling cost is spent. (~22k × 3 × 10 ≈ 660k requests — easily within Batch
  API limits.)
- **Tier 3 — stratified directness sample, N-sample self-consistency.** A
  fame/class-stratified sample of the broader population to measure the
  confidence gradient across arms (§7). Scale up toward full population if
  desired, since cost is not a constraint.

---

## 7. Re-running the directness analysis with continuous confidence

New questions the continuous signal unlocks (the scientific payoff):

1. **Confidence gradient across arms.** Does internal (self-consistency)
   confidence *drop* Baseline → Arm A → Arm B? A genuine drop = the inputs carry
   real signal (directness). Flat confidence = the model isn't using them.
2. **Calibration stratified by fame.** Is Arm B (anonymized name) confident *and*
   accurate only on famous (Q4) firms? That's memorization leakage caught as a
   per-stratum calibration gap — far stronger than an aggregate agreement rate.
3. **Confidence–accuracy coupling per arm.** A "direct" measurement has confidence
   that tracks accuracy. High confidence + flat accuracy in an arm = hollow
   confidence (memorized prior, not signal).

---

## 8. Threats to validity (discussion points)

1. **Sampling diversity on a reasoning model** (§3.4) — the make-or-break
   feasibility risk; pilot resolves it.
2. **Self-consistency confidence ≠ correctness.** Models can be *confidently
   wrong in unison*. The calibration analysis (§5) is what tests this — don't
   assume the signal is good, prove it.
3. **Gold-set noise.** Calibration metrics inherit any error in the gold labels;
   §4.3 (human κ check) bounds this.
4. **N and temperature are researcher choices** that affect the confidence values;
   report sensitivity analysis so results aren't artifacts of N=10.
5. **API nondeterminism** even with fixed settings; document the model snapshot
   and run dates.

---

## 9. Phased roadmap

| Phase | Output | Verify by |
|---|---|---|
| **0. Pilot probe** (~500 rows, 1 arm) | Confirm sampling produces real run-to-run diversity; pick N & temperature | Vote distributions are non-degenerate |
| **1. Gold-set refinement** | Frozen, versioned, stratified gold set + human-κ report | κ acceptable vs Tavily labels |
| **2. Tier-2 sampling run** | N-sample classifications on gold subset × 3 arms | Self-consistency confidence per row |
| **3. Calibration eval** | Reliability diagrams, ECE, Brier, AUROC, AURC — self-consistency vs verbalized | Self-consistency better-calibrated |
| **4. Tier-1/Tier-3 runs** | Full-pop single pass + stratified self-consistency sample | — |
| **5. Directness re-analysis** | Confidence gradient + fame-stratified calibration across arms | — |

---

## 10. Open questions for the advisor

- Is V2 a **replacement** for V1 or an **addition** (re-run both confidence
  methods on the same snapshot for a clean comparison)? Recommendation: addition.
- Acceptable inter-annotator agreement threshold for trusting the gold set?
- Do we want self-consistency on the **full** population (Tier 3 scaled up), or is
  a stratified sample sufficient for the directness gradient?
- Is dropping logprobs entirely acceptable, or do we want a parallel small-scale
  **GPT-4.1 (non-reasoning) logprob arm** as a methodological comparison point?

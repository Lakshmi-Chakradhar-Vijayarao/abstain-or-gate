# Pre-registration: abstention training vs runtime gating

**Status:** draft of 18 Nov 2026. Commit this file before the first training run; any later change is listed under Deviations with its date and reason.

## Question

At equal coverage, does training a model to abstain (QLoRA SFT, then DPO) beat a calibrated runtime gate, and do the two combine? A clear negative answer is a valid outcome.

## Design

| Element | Specification |
|---|---|
| Base model | Qwen2.5-3B-Instruct. **Deviation from the spec**, which asks for the model Project 1 serves (1–3B). gatedrag's measuring run uses 7B because 0.5B could not answer the corpus (pilot coverage 0.25). 3B keeps QLoRA on one T4 feasible. Every arm runs through gatedrag's pipeline, so the gate is the same code |
| Compute | One Kaggle T4 for training, two for serving; QLoRA (4-bit NF4). GPU-hours are recorded per seed in `run_info.json` |
| Training data | gatedrag's frozen **dev** split only. The builder refuses any test-split id (`data/pairs.py`) |
| Source of answers | An open-loop gatedrag pool of the base model on dev (K = 4 sampled answers per item), so wrong answers are not hidden by the gate |
| Pairs, type A | Unanswerable item, or an answerable item with no correct sample: chosen = abstain, rejected = a shipped (non-abstaining) answer |
| Pairs, type B | Answerable item with a correct sample: chosen = the shortest correct answer, rejected = abstain. These counter over-refusal |
| Pipeline | SFT (correct answers and justified abstentions), then DPO (A + B), each with LoRA r = 16, α = 32; DPO β = 0.1 |
| Seeds | 3 per trained arm (0, 1, 2) |

## Arms

| # | Arm | Purpose |
|---|---|---|
| 1 | Base | Reference |
| 2 | Base + abstention prompt (`abstain_strong`) | Cheapest alternative |
| 3 | SFT only | Isolates SFT's effect |
| 4 | SFT → DPO | The trained method |
| 5 | Base + gate **at equal coverage** | Runtime alternative: the base model's answers, gated by a support threshold chosen on **dev** so the gate answers as often as arm 4 does on dev. Derived from arm 1's runs, so it is paired item by item |
| 5c | Base + gatedrag's deployed cost gate (calibrator fitted on dev, c_wrong/c_abstain = 4) | The gate as it would ship |
| 6 | SFT → DPO + deployed cost gate | Do they combine? |

All arms use the same retrieval, prompt (except arm 2) and verifier. Arms 1–4 run open loop, so only the model's own abstentions count.

## Metrics

**Why matched coverage.** A fixed-threshold gate's score on most selective metrics is set by where its threshold falls: at c_wrong/c_abstain = 4 a 3B model rarely reaches p ≥ 0.75, so a deployed gate can sit near zero coverage. The design dry run showed exactly that. Comparing at equal coverage removes the threshold from the question.

**Primary ("replace?"):** selective accuracy, arm 4 vs arm 5, at arm 4's coverage.
- The coverage target is set on dev and applied to test unchanged.
- Paired bootstrap CI, resampling whole source documents (2,000 replicates), pooled over 3 seeds × K = 2 test runs.

**"Complement?":** expected cost per question at the stated costs (c_wrong = 4, c_abstain = 1), arm 6 vs arm 4 and arm 6 vs arm 5c, with the same paired CI.

**Secondary:**
- standard AURC for every arm that has its own ranking (arms 1–4; answered items ranked by verifier support). Gated arms report none, because they truncate another arm's ranking;
- selective accuracy at 70/80/90% coverage;
- over-abstention on answerable items;
- accuracy on answerable items.

**Transfer:** the same metrics on a held-out documentation corpus never used in training. The corpus and its golden set are to be chosen and frozen before training; listed under Deviations if this slips.

**Side effects:** lm-evaluation-harness on ARC-Easy and HellaSwag (first 500 items each, 0-shot) for base, SFT and SFT → DPO.

**Contamination:** 8-gram overlap and MiniLM cosine between training questions and test questions, reported whatever they show.

## Decision language, fixed now

- **Replace:** arm 4's selective accuracy at matched coverage is higher than arm 5's, and the CI excludes 0.
- **Complement:** arm 6's cost per question is lower than both arm 4's and arm 5c's, and both CIs exclude 0.
- **Both, either, or neither** are reported as such. Over-abstention is printed next to every verdict, so a "win" bought by refusing more is visible.

## Additions before the real run (19 Nov 2026, after an external review; no real-run data seen)

1. **Confidence-baseline arms**, at the same matched coverage as arm 5:
   - **arm 7: log-probability gate.** The base model's answers are ranked by mean token log-probability, with the threshold set on dev.
   - **arm 8: self-consistency gate.** The base model's answers are ranked by how many of 5 sampled answers agree with the most common one, with the threshold set on dev.

   SFT → DPO is compared with each, by selective accuracy and the same clustered paired bootstrap. If training cannot beat these cheap baselines, that is the result.
2. **Pair labels from the judge.** Correctness for building SFT data and preference pairs comes from the cross-family judge (Mistral-7B), not the token scorer. In gatedrag's human audit, the token scorer penalised correct paraphrases, which would turn answerable questions into "abstain" pairs.
3. **More pairs.** 8 sampled answers per dev item, and up to 3 distinct pairs per item and type. The pilot had only 38 type-B pairs.
4. **Secondary question: what does abstention DPO teach?**
   - The DPO log history (the chosen and rejected log-probabilities per step) is kept for every seed.
   - **Likelihood displacement**, where both fall, is reported next to over-abstention on answerable items. Together they distinguish a calibration gain from a learned refusal style.

## Labelling and reporting rules (19 Nov 2026)

- **Pre-revision pipeline:** any result from the Kaggle pilot started before the additions above (kernel version 8 and earlier) is labelled *pre-revision pipeline*. It has no confidence-baseline arms, uses token-scorer labels, and has one pair per item.
- **Judge reliability:** the pairs are labelled by the judge, so its error rate on wrong answers is reported with them. The rate is measured on gatedrag's wrong-enriched human batch (batch 3); on batch 1 the judge credited 4 of 9 wrong answers. This bounds how much bad signal can enter training: the share of type-B pairs whose "correct" answer is actually wrong is estimated from that rate.

## Deviations

- 18 Nov 2026: base model 3B rather than Project 1's served model (see Design).
- 18 Nov 2026, before any run: the primary comparison was changed from "AURC, arm 4 vs the deployed gate" to "selective accuracy at matched coverage". In the design dry run, the deployed gate's coverage, and so its AURC as then defined, was set by its threshold, not by its ranking. AURC is now the standard definition and is reported only for arms with their own ranking.
- **19 Nov 2026, after the real run (kernel version 9), labelling bug found: citation-only training targets.**
  - **What happened:** the judge graded 54 of 72 citation-only dev answers (e.g. "[1]") correct (corrected from 53 of 66, which missed comma-separated citations such as "[1], [2]"). "Shortest correct answer" then made a bare "[1]" the SFT target for 14 of 55 answerable items, and the chosen side of their type-B pairs.
  - **Effect:** the trained models answer with a bare citation far more often (seed 0: 0.27 of SFT and 0.26 of SFT → DPO test answers, against 0.09 for base).
  - **Version 9 still counts.** Its pre-registered verdict ("neither") is reported as the result of the protocol as written, with this bug named as its cause.
  - **Corrected re-run (version 10).** The only changes are:
    - citation-only answers are never correct targets (`pairs.has_content`);
    - the judge scores them wrong without a model call.
  - **Version 10 is reported beside version 9, not instead of it.** No other setting changes, and the decision language is unchanged.
- **19 Nov 2026, before kernel version 10's results were seen: the comparison is made at the common coverage.**
  - **Why:** a gate can only remove answers. In version 9, SFT → DPO answered more often on dev than base did on 2 of 3 seeds (dev coverage targets 0.649 and 0.644). Arm 5's threshold therefore became −∞, and "base + gate at matched coverage" was just the ungated base arm.
  - **Now:** both arms are compared at c = min(dev coverage of SFT → DPO, dev coverage of base). Whichever arm answers more is cut to c by a threshold on its own dev support; the other is unchanged. The same applies to arms 7 and 8. The decision language is unchanged, with "matched coverage" read as this common coverage.
  - **Version 9's comparisons are void** under either reading: on those 2 seeds they compared ungated arms.
- **Recorded after the fact (19 Nov 2026, found by an audit, not declared before the real run):**
  - **No transfer corpus was run.** The held-out documentation corpus was never chosen. Transfer is not reported.
  - **No MiniLM cosine.** The contamination report has only the 8-gram fields.
  - **No per-step DPO log.** Addition 4 promised chosen/rejected log-probabilities per step. The trainer logged one entry per seed, over 10 steps, so the displacement trajectory cannot be assessed; only end-of-training rewards are reported.
- **Known issues reported with every result:**
  - **The same question appears across the split under two roles.** Two security questions, `sec-005`/`sec-003` and `sec-001`/`sec-004`, appear in dev and test under different roles with opposite answerability. The splits are by document, so the frozen split is kept. The pair is reported as a memorisation-leak probe; future splits will deduplicate by question text.
  - **The pairs are unbalanced.** With judge labels, version 9 had 10 type-A against 144 type-B pairs. DPO therefore mostly teaches "answer rather than abstain". This is stated next to any DPO result, and the arms' coverage is read with it.

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

## Deviations

- 18 Nov 2026: base model 3B rather than Project 1's served model (see Design).
- 18 Nov 2026, before any run: the primary comparison was changed from "AURC, arm 4 vs the deployed gate" to "selective accuracy at matched coverage". In the design dry run, the deployed gate's coverage, and so its AURC as then defined, was set by its threshold, not by its ranking. AURC is now the standard definition and is reported only for arms with their own ranking.

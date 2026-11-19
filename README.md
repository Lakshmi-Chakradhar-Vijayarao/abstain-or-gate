# abstain-or-gate

**Is it better to train abstention into a model, gate it at runtime, or both?**

Nine arms on one support-docs QA task (the NATS documentation, through [gatedrag](../gated-llm-service)'s retrieval, prompt and verifier):

| # | Arm | Purpose |
|---|---|---|
| 1 | Base (Qwen2.5-3B-Instruct) | Reference |
| 2 | Base + abstention prompt | Cheapest alternative |
| 3 | QLoRA SFT | Isolates SFT |
| 4 | QLoRA SFT → DPO | The trained method |
| 5 | Base + gate at equal coverage (threshold set on dev) | Runtime alternative |
| 5c | Base + deployed cost gate | The gate as it would ship |
| 6 | SFT → DPO + gate | Do they combine? |
| 7 | Base + log-probability gate at equal coverage | Cheap confidence baseline |
| 8 | Base + self-consistency gate at equal coverage | Cheap confidence baseline |

> **Status: measured and locked (20 Nov 2026).** The result is in [docs/PREREG.md](docs/PREREG.md), "Result: kernel version 10".

## Result

Frozen split (87 dev, 247 test items), Qwen2.5-3B, 3 seeds. Arms are compared at equal coverage.

**Pre-registered verdict: neither.** Abstention training (QLoRA SFT → DPO) did not beat the runtime gate, and adding it to the gate did not lower cost.

| SFT → DPO minus … (selective accuracy) | Token scorer (primary) | LLM judge (check) |
|---|---|---|
| base + gate | +0.020 [−0.030, +0.071] | +0.045 [−0.000, +0.088] |
| base + log-probability gate | +0.023 [−0.031, +0.076] | +0.059 [+0.009, +0.103] |
| base + self-consistency gate | +0.030 [−0.025, +0.085] | +0.055 [+0.007, +0.102] |

**The scorers agree** that training ties the runtime gate. **They disagree** on the cheap confidence baselines: the judge gives training a small edge, the token scorer a tie. Both are reported, and the primary decides.

**On the way:**
- **The first full run was void.** The judge credited bare-citation answers ("[1]"), "shortest correct answer" made 14 of 55 training targets a bare citation, and the trained models learned to answer that way.
- **The matched-coverage comparison failed twice:** a gate can't raise coverage, and tied scores at the cut were all dropped.

All three were fixed and logged as deviations before the final run. Caveats are listed with the result: duplicated base runs, unbalanced DPO pairs, and no transfer corpus.

## How the training data is built

All training data comes from the **dev split**: the builder refuses any test item. It is built from an **open-loop** pool of the base model, where every sampled answer is shipped and scored, so wrong answers are not hidden by the gate.

| Item | SFT target | Preference pair |
|---|---|---|
| Answerable, some sample correct | that answer | **B**: chosen = correct answer, rejected = "I don't know." |
| Answerable, no sample correct | "I don't know." | **A**: chosen = abstain, rejected = a wrong shipped answer |
| Unanswerable | "I don't know." | **A**: chosen = abstain, rejected = a non-abstaining answer |

Type B pairs are what stop DPO from simply learning to refuse more. Over-abstention is reported next to every verdict for the same reason.

## Metrics

- **Primary:** selective accuracy at equal coverage. Both arms are cut to the lower of their two dev coverages, with tied scores split (`selective.cut_for_coverage`). **Combine:** cost per question at the stated costs.
- **Secondary:** selective accuracy at 70/80/90% coverage, over-abstention, and accuracy on answerable items.
- **Paired bootstrap CIs** resample whole source documents, over 3 seeds × 2 runs.
- **Also reported:** side effects (ARC-Easy and HellaSwag, 500 items each) and contamination (8-gram overlap between training and test questions; the planned embedding check was not run).

## Run

```bash
make install && make test
make dryrun          # whole pipeline on a laptop: fake generator + tiny model; checks wiring, produces no result
                     # (needs gatedrag cloned at ../gated-llm-service; builds its chunks on first run, which fetches the NATS docs)
# on Kaggle (T4): scripts/kaggle/run_project2.py with the gatedrag bundle and the frozen golden split attached
```

## Layout

```
src/abstaingate/
  data/        pairs.py (SFT + DPO data, dev only), prompts.py (exact gatedrag prompt)
  models/      train.py (QLoRA SFT -> merge -> DPO -> merge, GPU-hours logged)
  evaluation/  selective.py (AURC, sel-acc@coverage, clustered paired bootstrap),
               contamination.py, analysis.py (arm table + pre-registered verdict)
scripts/kaggle/run_project2.py   end-to-end run (resumable; DRY mode for laptops)
docs/PREREG.md
```

Apache-2.0.

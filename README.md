# abstain-or-gate

**Is it better to train abstention into a model, gate it at runtime, or both?**

Six arms on one support-docs QA task (the NATS documentation, through [gatedrag](../gated-llm-service)'s retrieval, prompt and verifier):

| # | Arm | Purpose |
|---|---|---|
| 1 | Base (Qwen2.5-3B-Instruct) | Reference |
| 2 | Base + abstention prompt | Cheapest alternative |
| 3 | QLoRA SFT | Isolates SFT |
| 4 | QLoRA SFT → DPO | The trained method |
| 5 | Base + gate at equal coverage (threshold set on dev) | Runtime alternative |
| 5c | Base + deployed cost gate | The gate as it would ship |
| 6 | SFT → DPO + gate | Do they combine? |

> **Status:** pipeline built and dry-run end to end; **no results yet**. The design, metrics, seeds and the wording of every possible verdict (replace / complement / neither) are pre-registered in [docs/PREREG.md](docs/PREREG.md). The run needs gatedrag's human-verified, frozen golden set and a Kaggle GPU.

## How the training data is built

All training data comes from the **dev split**: the builder refuses any test item. It is built from an **open-loop** pool of the base model, where every sampled answer is shipped and scored, so wrong answers are not hidden by the gate.

| Item | SFT target | Preference pair |
|---|---|---|
| Answerable, some sample correct | that answer | **B**: chosen = correct answer, rejected = "I don't know." |
| Answerable, no sample correct | "I don't know." | **A**: chosen = abstain, rejected = a wrong shipped answer |
| Unanswerable | "I don't know." | **A**: chosen = abstain, rejected = a non-abstaining answer |

Type B pairs are what stop DPO from simply learning to refuse more. Over-abstention is reported next to every verdict for the same reason.

## Metrics

- **Primary:** selective accuracy at matched coverage (SFT→DPO vs base + gate, gate threshold set on dev). **Combine:** cost per question at the stated costs.
- **Secondary:** selective accuracy at 70/80/90% coverage, over-abstention, and accuracy on answerable items.
- **Paired bootstrap CIs** resample whole source documents, over 3 seeds × 2 runs.
- **Also reported:** side effects (ARC-Easy and HellaSwag, 500 items each) and contamination (8-gram overlap and embedding similarity between training and test questions).

## Run

```bash
make install && make test
make dryrun          # whole pipeline on a laptop: fake generator + tiny model; checks wiring, produces no result
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

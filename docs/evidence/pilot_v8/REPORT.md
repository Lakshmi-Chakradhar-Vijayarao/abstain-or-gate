PILOT: golden items are unreviewed LLM drafts (manifest pilot_only); NOT a result.

| Arm | runs | coverage | selective_accuracy | aurc | cost_per_question | over_abstention |
|---|---|---|---|---|---|---|
| base | 6 | 0.598 | 0.569 | 0.266 | 1.433 | 0.253 |
| base_prompt | 6 | 0.557 | 0.615 | 0.290 | 1.301 | 0.303 |
| sft | 6 | 0.504 | 0.632 | 0.269 | 1.236 | 0.361 |
| sft_dpo | 6 | 0.513 | 0.634 | 0.275 | 1.238 | 0.348 |
| base_gate_deployed | 6 | 0.175 | 0.751 | — | 0.998 | 0.774 |
| sft_dpo_gate_deployed | 6 | 0.366 | 0.744 | — | 1.030 | 0.539 |
| base_gate_matched | 6 | 0.476 | 0.683 | — | 1.128 | 0.406 |

Replace? Selective accuracy, SFT→DPO − (base + gate at matched coverage) = -0.0485, 95% CI [-0.1058, +0.0027] (162 documents).
Complement? Cost per question, SFT→DPO + gate vs SFT→DPO: -0.2073, 95% CI [-0.2919, -0.1288].
Complement? Cost per question, SFT→DPO + gate vs base + gate: +0.0322, 95% CI [-0.0718, +0.1348].
Verdict: **neither**.

Contamination: {"n_train": 87, "n_test": 254, "ngram_n": 8, "ngram_max": 1.0, "ngram_mean": 0.009805676028464355, "ngram_flagged": 2, "exact_duplicates": 2}

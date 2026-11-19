| Arm | runs | coverage | selective_accuracy | aurc | cost_per_question | over_abstention |
|---|---|---|---|---|---|---|
| base | 6 | 0.593 | 0.628 | 0.188 | 1.289 | 0.226 |
| base_prompt | 6 | 0.563 | 0.672 | 0.235 | 1.174 | 0.266 |
| sft | 6 | 0.615 | 0.350 | 0.379 | 1.985 | 0.191 |
| sft_dpo | 6 | 0.633 | 0.363 | 0.380 | 1.981 | 0.168 |
| base_gate_deployed | 6 | 0.253 | 0.850 | — | 0.901 | 0.668 |
| sft_dpo_gate_deployed | 6 | 0.136 | 0.754 | — | 0.999 | 0.822 |
| base_gate_matched | 6 | 0.559 | 0.668 | — | 1.194 | 0.270 |
| base_logprob_matched | 6 | 0.579 | 0.637 | — | 1.263 | 0.245 |
| base_consistency_matched | 6 | 0.584 | 0.628 | — | 1.285 | 0.238 |

Replace? Selective accuracy, SFT→DPO − (base + gate at matched coverage) = -0.3006, 95% CI [-0.3621, -0.2379] (156 documents).
Complement? Cost per question, SFT→DPO + gate vs SFT→DPO: -0.9818, 95% CI [-1.1433, -0.8100].
Complement? Cost per question, SFT→DPO + gate vs base + gate: +0.0985, 95% CI [+0.0039, +0.1797].
Verdict: **neither**.

Contamination: {"n_train": 87, "n_test": 247, "ngram_n": 8, "ngram_max": 1.0, "ngram_mean": 0.009853536556322933, "ngram_flagged": 2, "exact_duplicates": 2}

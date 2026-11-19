| Arm | runs | coverage | selective_accuracy | aurc | cost_per_question | over_abstention |
|---|---|---|---|---|---|---|
| base | 6 | 0.593 | 0.628 | 0.188 | 1.289 | 0.226 |
| base_prompt | 6 | 0.563 | 0.672 | 0.235 | 1.174 | 0.266 |
| sft | 6 | 0.588 | 0.637 | 0.237 | 1.273 | 0.230 |
| sft_dpo | 6 | 0.625 | 0.636 | 0.238 | 1.296 | 0.188 |
| base_gate_deployed | 6 | 0.253 | 0.850 | — | 0.901 | 0.668 |
| sft_dpo_gate_deployed | 6 | 0.202 | 0.768 | — | 0.984 | 0.734 |
| base_gate_matched | 6 | 0.524 | 0.708 | — | 1.099 | 0.313 |
| base_logprob_matched | 6 | 0.580 | 0.633 | — | 1.271 | 0.244 |
| base_consistency_matched | 6 | 0.574 | 0.627 | — | 1.281 | 0.251 |

Replace? Selective accuracy, SFT→DPO − (base + gate at matched coverage) = -0.0710, 95% CI [-0.1187, -0.0204] (156 documents).
Complement? Cost per question, SFT→DPO + gate vs SFT→DPO: -0.3117, 95% CI [-0.4399, -0.1887].
Complement? Cost per question, SFT→DPO + gate vs base + gate: +0.0830, 95% CI [-0.0140, +0.1660].
Verdict: **neither**.

Contamination: {"n_train": 87, "n_test": 247, "ngram_n": 8, "ngram_max": 1.0, "ngram_mean": 0.009853536556322933, "ngram_flagged": 2, "exact_duplicates": 2}

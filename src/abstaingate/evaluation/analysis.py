"""Arm table and the pre-registered comparisons (docs/PREREG.md), from gatedrag pools.

Pools on disk (from scripts/kaggle/run_project2.py), per seed s:
  test_base_seed{s}, test_base_prompt_seed{s}, test_sft_seed{s}, test_sft_dpo_seed{s}   open loop
  test_base_gate_seed{s}, test_sft_dpo_gate_seed{s}      the deployed cost gate (c_wrong/c_abstain = 4)
  dev_base_openloop, dev_sft_dpo_openloop_seed{s}         dev pools, used only to set thresholds

Arm 5 ("base + gate at equal coverage") is derived, not run: the base model's test answers, gated
by a support threshold chosen on DEV so the gate answers as often as SFT -> DPO does on dev.

  replace?     selective accuracy, arm 4 (SFT -> DPO) vs arm 5, at that matched coverage
  complement?  cost per question at the stated costs, arm 6 (SFT -> DPO + deployed gate) vs
               arm 4 and vs arm 5c (base + deployed gate)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from abstaingate.evaluation.selective import (
    cost_per_question,
    coverage,
    gate,
    paired_cluster_bootstrap,
    report,
    rows_from_pool,
    selective_accuracy,
    threshold_for_coverage,
)

RUN_ARMS = {"base": "test_base", "base_prompt": "test_base_prompt", "sft": "test_sft", "sft_dpo": "test_sft_dpo",
            "base_gate_deployed": "test_base_gate", "sft_dpo_gate_deployed": "test_sft_dpo_gate"}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _runs(pool: dict) -> list[list[dict]]:
    return [rows_from_pool(pool, r) for r in range(len(pool["runs"]))]


def _flat(pool: dict | None) -> list[dict]:
    return [r for rows in _runs(pool) for r in rows] if pool else []


def analyse(pool_dir: str | Path, n_boot: int = 2000) -> dict:
    d = Path(pool_dir)
    seeds = sorted({int(p.stem.rsplit("seed", 1)[1]) for p in d.glob("test_*_seed*.json")})
    arms: dict[str, list[list[dict]]] = {}          # arm -> list of row-sets, one per (seed, run)
    thresholds = {}
    dev_base = _flat(_load(d / "dev_base_openloop.json"))
    for s in seeds:
        for arm, prefix in RUN_ARMS.items():
            p = _load(d / f"{prefix}_seed{s}.json")
            if p:
                arms.setdefault(arm, []).extend(_runs(p))
        dev_dpo = _flat(_load(d / f"dev_sft_dpo_openloop_seed{s}.json"))
        base = _load(d / f"test_base_seed{s}.json")
        if dev_base and dev_dpo and base:
            target = coverage(dev_dpo)
            thr = threshold_for_coverage(dev_base, target)
            thresholds[s] = {"dev_coverage_target": target, "support_threshold": thr}
            arms.setdefault("base_gate_matched", []).extend(gate(rows, thr) for rows in _runs(base))

    keys = ["coverage", "selective_accuracy", "aurc", "cost_per_question", "sel_acc@70%", "sel_acc@80%",
            "sel_acc@90%", "over_abstention", "accuracy_answerable"]

    def pooled(rows_sets, k):
        vals = [v for v in (report(rs)[k] for rs in rows_sets) if v is not None and v == v]
        return float(np.mean(vals)) if vals else None

    table = {arm: {"n_runs": len(rs), **{k: pooled(rs, k) for k in keys}} for arm, rs in arms.items()}
    for arm in table:            # a gated arm truncates another arm's ranking: AURC belongs to that arm
        if "gate" in arm:
            table[arm]["aurc"] = None

    def compare(a: str, b: str, stat) -> dict | None:
        if a not in arms or b not in arms:
            return None
        n = min(len(arms[a]), len(arms[b]))
        fa = [{**r, "item": f"{k}:{r['item']}"} for k, rs in enumerate(arms[a][:n]) for r in rs]
        fb = [{**r, "item": f"{k}:{r['item']}"} for k, rs in enumerate(arms[b][:n]) for r in rs]
        return paired_cluster_bootstrap(fa, fb, stat=stat, n_boot=n_boot)

    replace = compare("sft_dpo", "base_gate_matched", selective_accuracy)
    c64 = compare("sft_dpo_gate_deployed", "sft_dpo", cost_per_question)
    c65 = compare("sft_dpo_gate_deployed", "base_gate_deployed", cost_per_question)
    verdict: dict = {
        "replace": None if replace is None else bool(replace["ci"][0] > 0),
        "complement": None if (c64 is None or c65 is None) else bool(c64["ci"][1] < 0 and c65["ci"][1] < 0),
    }
    verdict["summary"] = ("pending" if None in verdict.values() else
                          "replace and complement" if verdict["replace"] and verdict["complement"] else
                          "replace" if verdict["replace"] else "complement" if verdict["complement"] else "neither")
    return {"table": table, "thresholds": thresholds,
            "replace_selacc_sft_dpo_vs_matched_gate": replace,
            "complement_cost_vs_sft_dpo": c64, "complement_cost_vs_base_gate": c65, "verdict": verdict}


def render(res: dict) -> str:
    cols = ["coverage", "selective_accuracy", "aurc", "cost_per_question", "over_abstention"]
    lines = ["| Arm | runs | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for arm, row in res["table"].items():
        cells = ["—" if row[c] is None else f"{row[c]:.3f}" for c in cols]
        lines.append(f"| {arm} | {row['n_runs']} | " + " | ".join(cells) + " |")
    r = res["replace_selacc_sft_dpo_vs_matched_gate"]
    if r:
        lines += ["", f"Replace? Selective accuracy, SFT→DPO − (base + gate at matched coverage) = {r['diff']:+.4f}, "
                      f"95% CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] ({r['n_clusters']} documents)."]
    for key, label in (("complement_cost_vs_sft_dpo", "vs SFT→DPO"), ("complement_cost_vs_base_gate", "vs base + gate")):
        c = res[key]
        if c:
            lines.append(f"Complement? Cost per question, SFT→DPO + gate {label}: {c['diff']:+.4f}, "
                         f"95% CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}].")
    lines.append(f"Verdict: **{res['verdict']['summary']}**.")
    return "\n".join(lines) + "\n"

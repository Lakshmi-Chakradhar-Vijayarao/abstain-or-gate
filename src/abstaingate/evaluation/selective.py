"""Selective-prediction metrics, defined so a gate's threshold cannot decide the comparison.

Per item (one sampled answer): `answered` (the model did not abstain), `correct` (answerable:
shipped answer correct; unanswerable: abstained), `conf` (gatedrag's verifier support, the same
yardstick for every arm), `answerable`, `cluster` (source document), `item`.

  coverage              share of items answered
  selective accuracy    correct among answered
  AURC (standard)       answered items ranked by conf; the mean, over k = 1..n_answered, of the
                        error rate among the top k. A gate that only truncates this ranking does
                        not change it, so AURC compares rankings, not thresholds.
  gate(rows, thr)       a runtime gate: answered only if the model answered AND conf >= thr
  threshold_for_coverage(dev_rows, c)   the support threshold at which a gate answers share c of
                        dev items: how "equal coverage" is set without looking at test
  cost per question     c_wrong * wrong-and-shipped + c_abstain * abstained, per item (stated costs)
"""

from __future__ import annotations

import math

import numpy as np


def _arrays(rows: list[dict]):
    answered = np.array([r["answered"] for r in rows], bool)
    correct = np.array([r["correct"] for r in rows], float)
    conf = np.array([r["conf"] for r in rows], float)
    return answered, correct, conf


def coverage(rows: list[dict]) -> float:
    return float(np.mean([r["answered"] for r in rows])) if rows else float("nan")


def selective_accuracy(rows: list[dict]) -> float:
    answered, correct, _ = _arrays(rows)
    return float(correct[answered].mean()) if answered.any() else float("nan")


def aurc(rows: list[dict]) -> float:
    answered, correct, conf = _arrays(rows)
    if not answered.any():
        return float("nan")
    order = np.argsort(-conf[answered], kind="stable")
    err = 1.0 - correct[answered][order]
    return float((np.cumsum(err) / np.arange(1, err.size + 1)).mean())


def gate(rows: list[dict], threshold: float) -> list[dict]:
    """Apply a support threshold to answers the model gave; a gated-out item scores like the
    model's own abstention (correct only on unanswerable items)."""
    out = []
    for r in rows:
        keep = bool(r["answered"] and r["conf"] >= threshold)
        out.append({**r, "answered": keep,
                    "correct": r["correct"] if keep else (0.0 if r["answerable"] else 1.0)})
    return out


def threshold_for_coverage(rows: list[dict], target: float) -> float:
    """A support threshold whose gate answers about `target` of the items (never more)."""
    confs = sorted((r["conf"] for r in rows if r["answered"]), reverse=True)
    k = math.floor(target * len(rows))
    if k >= len(confs):
        return -math.inf                          # the model never answers that much: no gating needed
    if k == 0:
        return math.inf
    return float(np.nextafter(confs[k], math.inf))


def selective_accuracy_at(rows: list[dict], cov: float) -> float | None:
    """Selective accuracy when the arm's own answers are ranked by conf and cut at coverage `cov`."""
    answered, correct, conf = _arrays(rows)
    k = math.ceil(cov * len(rows))
    if k == 0 or answered.sum() < k:
        return None
    idx = np.flatnonzero(answered)[np.argsort(-conf[answered], kind="stable")][:k]
    return float(correct[idx].mean())


def cost_per_question(rows: list[dict], c_wrong: float = 4.0, c_abstain: float = 1.0) -> float:
    answered, correct, _ = _arrays(rows)
    cost = np.where(answered, (1.0 - correct) * c_wrong, c_abstain)
    return float(cost.mean())


def report(rows: list[dict], levels=(0.7, 0.8, 0.9)) -> dict:
    answered, correct, _ = _arrays(rows)
    ans = np.array([r["answerable"] for r in rows], bool)
    return {
        "n": len(rows), "coverage": coverage(rows), "selective_accuracy": selective_accuracy(rows),
        "aurc": aurc(rows), "cost_per_question": cost_per_question(rows),
        **{f"sel_acc@{c:.0%}": selective_accuracy_at(rows, c) for c in levels},
        "over_abstention": float((~answered[ans]).mean()) if ans.any() else None,
        "accuracy_answerable": float(correct[ans].mean()) if ans.any() else None,
    }


def paired_cluster_bootstrap(rows_a: list[dict], rows_b: list[dict], stat=aurc, n_boot: int = 2000,
                             seed: int = 0, alpha: float = 0.05) -> dict:
    """CI for stat(a) - stat(b). Rows are paired by position (same items, same order); whole
    source documents are resampled, so items from one document move together."""
    if len(rows_a) != len(rows_b) or any(x["item"] != y["item"] for x, y in zip(rows_a, rows_b, strict=True)):
        raise ValueError("arms must be scored on the same items in the same order")
    clusters: dict[str, list[int]] = {}
    for i, r in enumerate(rows_a):
        clusters.setdefault(r["cluster"], []).append(i)
    keys = list(clusters)
    rng = np.random.default_rng(seed)
    point = stat(rows_a) - stat(rows_b)
    diffs = []
    for _ in range(n_boot):
        idx = [i for k in rng.choice(len(keys), len(keys), replace=True) for i in clusters[keys[k]]]
        d = stat([rows_a[i] for i in idx]) - stat([rows_b[i] for i in idx])
        if d == d:                                # skip resamples where a statistic is undefined
            diffs.append(d)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2]) if diffs else (float("nan"), float("nan"))
    return {"diff": point, "ci": [float(lo), float(hi)], "n_clusters": len(keys), "n_boot": len(diffs)}


def rows_from_pool(pool: dict, run: int = 0) -> list[dict]:
    """One row per item from one run of a gatedrag pool."""
    ids, clusters = pool["meta"]["item_ids"], pool["meta"]["item_clusters"]
    return [{"item": i, "cluster": c, "answered": sc["abstained"] == 0.0, "correct": sc["correct"],
             "conf": sc["support"], "answerable": sc["answerable"] == 1.0}
            for i, c, sc in zip(ids, clusters, pool["runs"][run], strict=True)]

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


def _tie_draw(item: str) -> float:
    """A fixed pseudo-random number in [0, 1) per item, for splitting ties reproducibly."""
    import hashlib
    return int(hashlib.md5(item.encode()).hexdigest()[:8], 16) / 2 ** 32


def gate(rows: list[dict], threshold: float, tie_keep: float = 1.0) -> list[dict]:
    """Apply a support threshold to answers the model gave; a gated-out item scores like the
    model's own abstention (correct only on unanswerable items). Answers with conf > threshold are
    kept; answers with conf == threshold are kept for the share `tie_keep` of items (chosen by a
    fixed hash of the item id), so a coverage target can be met even when many scores tie."""
    out = []
    for r in rows:
        keep = bool(r["answered"] and (r["conf"] > threshold or
                                       (r["conf"] == threshold and _tie_draw(r["item"]) < tie_keep)))
        out.append({**r, "answered": keep,
                    "correct": r["correct"] if keep else (0.0 if r["answerable"] else 1.0)})
    return out


def cut_for_coverage(rows: list[dict], target: float) -> tuple[float, float]:
    """(threshold, tie_keep) for `gate` so that it answers about `target` of the items: every answer
    scored above the threshold, plus the share `tie_keep` of those scored exactly at it."""
    confs = sorted((r["conf"] for r in rows if r["answered"]), reverse=True)
    m = math.floor(target * len(rows))
    if m >= len(confs):
        return -math.inf, 1.0                       # the model never answers that much: no gating needed
    if m == 0:
        return math.inf, 0.0
    v = confs[m - 1]
    above = sum(c > v for c in confs)
    tied = sum(c == v for c in confs)
    return float(v), (m - above) / tied


def threshold_for_coverage(rows: list[dict], target: float) -> float:
    """A support threshold whose gate answers about `target` of the items (never more). Ties at the
    cut are dropped entirely, so with many tied scores coverage can fall far below the target; use
    `cut_for_coverage` for comparisons at matched coverage."""
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


def rows_from_pool(pool: dict, run: int = 0, conf: str = "support", consistency: dict | None = None) -> list[dict]:
    """One row per item from one run of a gatedrag pool. `conf` picks the ranking score:
      support      gatedrag's verifier support (the gate's score)
      logprob      the answer's mean token log-probability (a model-confidence baseline)
      consistency  agreement among K sampled answers to the item (`consistency` maps item -> score)"""
    ids, clusters = pool["meta"]["item_ids"], pool["meta"]["item_clusters"]
    recs = pool.get("records", [[{}] * len(ids)] * len(pool["runs"]))[run]
    rows = []
    for i, c, sc, rec in zip(ids, clusters, pool["runs"][run], recs, strict=True):
        if conf == "support":
            score = sc["support"]
        elif conf == "logprob":
            lp = rec.get("mean_logprob")
            score = float("-inf") if lp is None else float(lp)
        else:
            score = (consistency or {}).get(i, 0.0)
        rows.append({"item": i, "cluster": c, "answered": sc["abstained"] == 0.0, "correct": sc["correct"],
                     "conf": score, "answerable": sc["answerable"] == 1.0})
    return rows


def consistency_scores(pool: dict, min_overlap: float = 0.6) -> dict[str, float]:
    """Self-consistency per item: the share of the item's K sampled answers that agree with the most
    common one (agreement = token overlap >= min_overlap, both ways). Abstentions count as their own answer."""
    import re as _re

    def toks(t: str) -> set[str]:
        return set(_re.findall(r"[a-z0-9]+", t.lower()))

    def agree(a: str, b: str) -> bool:
        ta, tb = toks(a), toks(b)
        if not ta or not tb:
            return ta == tb
        inter = len(ta & tb)
        return inter / len(ta) >= min_overlap and inter / len(tb) >= min_overlap

    out = {}
    ids = pool["meta"]["item_ids"]
    for j, item in enumerate(ids):
        answers = [run[j]["answer"] for run in pool["records"]]
        best = max(sum(agree(a, b) for b in answers) for a in answers)
        out[item] = best / len(answers)
    return out

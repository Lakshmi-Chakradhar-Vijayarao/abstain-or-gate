"""Build SFT examples and DPO preference pairs from gatedrag's DEV split only.

Input: an open-loop gatedrag pool (router = open_loop: every generated answer shipped and scored,
with the answer text and the retrieved chunk ids stored), the dev split it was run on, and the
chunk file, so each prompt is rebuilt exactly as the model saw it.

Per item (K sampled answers from the base model):

  answerable, some answer correct      SFT: the shortest correct answer
                                        pair B: chosen = that answer, rejected = abstain
  answerable, no answer correct        SFT: abstain ("unsupported": the passages did not let the
                                        base model answer correctly)
                                        pair A: chosen = abstain, rejected = a wrong shipped answer
  unanswerable                          SFT: abstain
                                        pair A: chosen = abstain, rejected = a non-abstaining answer

Type B pairs exist to counter over-refusal: with type A alone, DPO tends to learn "refuse more".
The test split is never read here; `build` refuses any item id that is in the test manifest.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ABSTAIN = "I don't know."


@dataclass
class Built:
    sft: list[dict]
    pairs: list[dict]
    counts: dict


def has_content(text: str) -> bool:
    """False for an answer that is only citation markers and punctuation ("[1]"). The judge grades most
    of these CORRECT (19 Nov 2026: 53 of 66 in the dev pool), and "shortest correct answer" then picks
    them, so 13 of 55 SFT answer targets were a bare "[1]". Such answers are never a correct target;
    they stay available as rejected (wrong, shipped) answers."""
    return re.search(r"[A-Za-z0-9]", re.sub(r"\[\d+\]", "", text)) is not None


def _is_abstain(text: str) -> bool:
    return text.strip().lower().rstrip(".") in ("i don't know", "i do not know", "")


def build(pool: dict, items: list[dict], prompt_for, test_ids: set[str] | frozenset = frozenset(),
          label: str = "correct", max_pairs_per_item: int = 1) -> Built:
    """`prompt_for(item, retrieved_chunk_ids) -> list[messages]` rebuilds the exact prompt.
    `items` are golden dicts (dev split). Raises if any item is in `test_ids`.
    `label` is the score that decides correctness: "correct" (token scorer) or "correct_judge" (the
    LLM judge; a human audit found the token scorer penalises correct paraphrases, which would turn
    answerable questions into type-A "abstain" pairs). `max_pairs_per_item` > 1 uses several distinct
    sampled answers per item, so the pair count is not capped at one per question."""
    leak = [i["id"] for i in items if i["id"] in test_ids]
    if leak:
        raise ValueError(f"{len(leak)} test-split item(s) passed to the training-data builder: {leak[:5]}")
    if pool["meta"]["config"].get("router") != "open_loop":
        raise ValueError("build training data from an open-loop pool, so wrong answers are not hidden by the gate")
    if "records" not in pool:
        raise ValueError("pool has no answer records")
    by_id = {it["id"]: it for it in items}
    ids = pool["meta"]["item_ids"]
    per_item: dict[str, list[tuple[dict, dict]]] = {}
    for scores, recs in zip(pool["runs"], pool["records"], strict=True):
        for item_id, sc, rec in zip(ids, scores, recs, strict=True):
            if item_id in by_id:
                per_item.setdefault(item_id, []).append((sc, rec))
    sft, pairs = [], []
    counts = {"items": 0, "sft_answer": 0, "sft_abstain": 0, "pair_A": 0, "pair_B": 0, "no_pair": 0}
    for item_id, samples in per_item.items():
        it = by_id[item_id]
        counts["items"] += 1
        retrieved = samples[0][1].get("retrieved") or []
        prompt = prompt_for(it, retrieved)
        meta = {"item": item_id, "cluster": it.get("cluster") or (it.get("gold_docs") or [item_id])[0]}
        shipped = [rec["answer"] for _, rec in samples if not _is_abstain(rec["answer"])]
        if it.get("answerable"):
            correct = sorted({rec["answer"] for sc, rec in samples if sc.get(label, sc["correct"]) == 1.0
                              and not _is_abstain(rec["answer"]) and has_content(rec["answer"])}, key=len)
            if correct:
                sft.append({"prompt": prompt, "completion": correct[0], **meta})
                for ans in correct[:max_pairs_per_item]:
                    pairs.append({"prompt": prompt, "chosen": ans, "rejected": ABSTAIN, "type": "B", **meta})
                    counts["pair_B"] += 1
                counts["sft_answer"] += 1
                continue
            sft.append({"prompt": prompt, "completion": ABSTAIN, **meta})
            counts["sft_abstain"] += 1
        else:
            sft.append({"prompt": prompt, "completion": ABSTAIN, **meta})
            counts["sft_abstain"] += 1
        if shipped:
            for ans in sorted(set(shipped), key=len)[:max_pairs_per_item]:
                pairs.append({"prompt": prompt, "chosen": ABSTAIN, "rejected": ans, "type": "A", **meta})
                counts["pair_A"] += 1
        else:
            counts["no_pair"] += 1
    return Built(sft, pairs, counts)


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

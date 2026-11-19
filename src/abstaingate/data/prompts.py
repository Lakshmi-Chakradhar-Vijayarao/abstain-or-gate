"""Rebuild the exact prompt a gatedrag answer was generated from (same system prompt, same passages)."""

from __future__ import annotations


def prompt_builder(chunks_path: str, prompt_variant: str = "grounded"):
    from gatedrag.data.corpus import load_chunks
    from gatedrag.models.generation import build_messages
    from gatedrag.models.retrieval import Hit

    by_id = {c.chunk_id: c for c in load_chunks(chunks_path)}

    def prompt_for(item: dict, retrieved: list[str]) -> list[dict]:
        hits = [Hit(by_id[cid], 0.0, r) for r, cid in enumerate(retrieved) if cid in by_id]
        return build_messages(item["question"], hits, prompt_variant)

    return prompt_for

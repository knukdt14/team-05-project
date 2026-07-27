"""Diverse slide retrieval selected by the embedding/retrieval experiments."""

from __future__ import annotations

from typing import Any


def retrieve_diverse(
    collection,
    embedder,
    query: str,
    *,
    k: int = 5,
    fetch_k: int = 15,
) -> list[dict[str, Any]]:
    if fetch_k < k:
        raise ValueError("fetch_k는 k 이상이어야 합니다.")

    result = collection.query(
        query_embeddings=[embedder.embed_query(query)],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )
    selected = []
    seen_slides = set()
    for chunk_id, text, metadata, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        slide_no = int(metadata["slide_no"])
        if slide_no in seen_slides:
            continue
        seen_slides.add(slide_no)
        selected.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "slide_no": slide_no,
                "title": metadata.get("title", ""),
                "distance": float(distance),
            }
        )
        if len(selected) == k:
            break
    return selected


def recall_at_k(retrieved: list[int], relevant: set[int]) -> float:
    return len(set(retrieved) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    for rank, value in enumerate(retrieved, start=1):
        if value in relevant:
            return 1.0 / rank
    return 0.0


def context_precision(retrieved: list[int], relevant: set[int]) -> float:
    return len(set(retrieved) & relevant) / len(retrieved) if retrieved else 0.0


def diversity_at_k(retrieved: list[int], k: int) -> float:
    return len(set(retrieved)) / k if k else 0.0


"""Chroma index construction shared by selection and quiz generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb


def build_index(
    chunks: list[dict[str, Any]],
    embedder,
    *,
    collection_name: str,
    persist_dir: str | Path | None = None,
    space: str = "l2",
):
    if persist_dir is None:
        client = chromadb.EphemeralClient()
    else:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        collection_name,
        metadata={"hnsw:space": space},
    )

    texts = [chunk["text"] for chunk in chunks]
    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        embeddings=embedder.embed_documents(texts),
        documents=texts,
        metadatas=[
            {
                "slide_no": int(chunk["slide_no"]),
                "chunk_method": str(chunk.get("chunk_method", "")),
                "title": str(chunk.get("title", "")),
            }
            for chunk in chunks
        ],
    )
    return collection


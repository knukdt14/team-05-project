"""PowerPoint chunking implementations used by the quiz generator."""

from __future__ import annotations

import re
from typing import Any, Iterable


SUPPORTED_STRATEGIES = (
    "recursive",
    "sentence_pack",
    "slide_aware",
    "title_body",
)
CHUNKING_CONFIGS = {
    "recursive": {"chunk_size": 150, "overlap": 30},
    "sentence_pack": {"chunk_size": 300, "overlap": 1},
    "slide_aware": {"chunk_size": 300, "overlap": 1},
    "title_body": {"chunk_size": 300, "overlap": 1},
}
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def sentence_units(text: str) -> list[str]:
    units = []
    for line in _lines(text):
        units.extend(
            part.strip()
            for part in SENTENCE_BOUNDARY.split(line)
            if part.strip()
        )
    return units


def _hard_split(text: str, size: int) -> list[str]:
    return [text[start : start + size] for start in range(0, len(text), size)]


def _pack(units: Iterable[str], size: int, overlap_units: int) -> list[str]:
    normalized = []
    for unit in units:
        clean = unit.strip()
        if not clean:
            continue
        normalized.extend([clean] if len(clean) <= size else _hard_split(clean, size))

    chunks: list[str] = []
    current: list[str] = []
    for unit in normalized:
        if current and len("\n".join([*current, unit])) > size:
            chunks.append("\n".join(current))
            current = current[-overlap_units:] if overlap_units else []
            while current and len("\n".join([*current, unit])) > size:
                current.pop(0)
        current.append(unit)
    if current:
        final = "\n".join(current)
        if not chunks or chunks[-1] != final:
            chunks.append(final)
    return chunks


def recursive_chunk(text: str, size: int = 150, overlap: int = 30) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("recursive 설정은 size > 0, 0 <= overlap < size여야 합니다.")
    chunks = _pack(sentence_units(text), size, 0)
    return [
        f"{chunks[index - 1][-overlap:]} {chunk}" if index and overlap else chunk
        for index, chunk in enumerate(chunks)
    ]


def sentence_pack_chunk(text: str, size: int = 300, overlap: int = 1) -> list[str]:
    return _pack(sentence_units(text), size, overlap)


def _title_body(text: str) -> tuple[str, str]:
    lines = _lines(text)
    return (lines[0], "\n".join(lines[1:])) if lines else ("", "")


def _format(title: str, body: str) -> str:
    return "\n".join(
        value for value in [f"[제목] {title}" if title else "", body] if value
    )


def _pieces_for_document(
    document: dict[str, Any],
    strategy: str,
    size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    text = str(document.get("text", ""))
    if strategy == "recursive":
        return [{"text": value} for value in recursive_chunk(text, size, overlap)]
    if strategy == "sentence_pack":
        return [{"text": value} for value in sentence_pack_chunk(text, size, overlap)]

    if strategy == "slide_aware":
        title, body = _title_body(text)
    else:
        title = str(document.get("title", "")).strip()
        groups = document.get("body_groups") or []
        body = "\n".join(
            str(group.get("text", "")).strip()
            for group in groups
            if str(group.get("text", "")).strip()
        )
        if not body:
            _, body = _title_body(text)

    if not title and not body:
        return []
    bodies = (
        [body]
        if len(body) <= size
        else sentence_pack_chunk(body, size=size, overlap=overlap)
    )
    return [{"text": _format(title, value), "title": title} for value in bodies or [""]]


def chunk_documents(
    documents: list[dict[str, Any]],
    strategy: str,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
    document_id: str = "ajin_training_250416",
) -> list[dict[str, Any]]:
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"지원하지 않는 전략: {strategy}")
    config = CHUNKING_CONFIGS[strategy]
    size = config["chunk_size"] if chunk_size is None else chunk_size
    overlap_value = config["overlap"] if overlap is None else overlap

    chunks = []
    for document in documents:
        slide_no = int(document["slide_no"])
        pieces = _pieces_for_document(
            document,
            strategy,
            size,
            overlap_value,
        )
        for index, piece in enumerate(pieces, start=1):
            text = piece["text"].strip()
            if not text:
                continue
            chunks.append(
                {
                    "chunk_id": f"{document_id}-p{slide_no:03d}-c{index:02d}",
                    "slide_no": slide_no,
                    "slide_chunk_index": index - 1,
                    "text": text,
                    "title": piece.get("title", document.get("title", "")),
                    "chunk_method": strategy,
                    "chunk_size": size,
                    "overlap": overlap_value,
                }
            )
    return chunks

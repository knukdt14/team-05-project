"""Comparable chunking strategies for PowerPoint-based RAG.

Strategies
----------
recursive
    Existing baseline: 150 characters with 30-character overlap.
sentence_pack
    Pack complete sentences/lines up to 300 characters and overlap one unit.
slide_aware
    Use the original flat slide text, repeat the first line as a title, and
    split only inside a slide at line/sentence boundaries.
title_body
    Use structured PPT extraction to create a title-body group, then split only
    a long body at line/sentence boundaries.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


DEFAULT_STRATEGY = "recursive"
CHUNK_SIZE = 150
OVERLAP = 30
SENTENCE_PACK_SIZE = 300
SENTENCE_OVERLAP = 1
SLIDE_AWARE_SIZE = 300
SUPPORTED_STRATEGIES = (
    "recursive",
    "sentence_pack",
    "slide_aware",
    "title_body",
)

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+")


def _non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _split_line_into_sentences(line: str) -> list[str]:
    """Split a line at sentence punctuation, retaining bullet lines as units."""
    sentences = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(line) if part.strip()]
    return sentences or ([line.strip()] if line.strip() else [])


def sentence_units(text: str) -> list[str]:
    """Return PPT-friendly units: line boundaries first, sentence boundaries second."""
    units: list[str] = []
    for line in _non_empty_lines(text):
        units.extend(_split_line_into_sentences(line))
    return units


def _hard_split(text: str, chunk_size: int) -> list[str]:
    return [text[start : start + chunk_size] for start in range(0, len(text), chunk_size)]


def recursive_chunk(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[str]:
    """Existing recursive-character baseline, kept for fair comparison."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    paragraphs = _non_empty_lines(text)
    chunks: list[str] = []
    buffer = ""

    for paragraph in paragraphs:
        candidate = f"{buffer}\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue

        if buffer:
            chunks.append(buffer)

        if len(paragraph) > chunk_size:
            chunks.extend(_pack_units(sentence_units(paragraph), chunk_size, overlap_units=0))
            buffer = ""
        else:
            buffer = paragraph

    if buffer:
        chunks.append(buffer)

    if overlap == 0:
        return chunks

    overlapped: list[str] = []
    for index, chunk in enumerate(chunks):
        if index > 0:
            chunk = f"{chunks[index - 1][-overlap:]} {chunk}"
        overlapped.append(chunk)
    return overlapped


def _pack_units(
    units: Iterable[str],
    chunk_size: int,
    overlap_units: int,
) -> list[str]:
    """Pack complete units up to a character budget with unit-level overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap_units < 0:
        raise ValueError("overlap_units must be non-negative")

    normalized: list[str] = []
    for unit in units:
        clean = unit.strip()
        if not clean:
            continue
        if len(clean) <= chunk_size:
            normalized.append(clean)
        else:
            normalized.extend(_hard_split(clean, chunk_size))

    chunks: list[str] = []
    current: list[str] = []

    for unit in normalized:
        candidate = "\n".join([*current, unit])
        if current and len(candidate) > chunk_size:
            chunks.append("\n".join(current))
            current = current[-overlap_units:] if overlap_units else []

            # Avoid a non-progressing loop when the overlapped unit plus the new
            # unit is still too large.
            while current and len("\n".join([*current, unit])) > chunk_size:
                current.pop(0)

        current.append(unit)

    if current:
        final = "\n".join(current)
        if not chunks or final != chunks[-1]:
            chunks.append(final)

    return chunks


def sentence_pack_chunk(
    text: str,
    chunk_size: int = SENTENCE_PACK_SIZE,
    overlap_sentences: int = SENTENCE_OVERLAP,
) -> list[str]:
    """Pack sentences/PPT lines and overlap the previous complete unit."""
    return _pack_units(
        sentence_units(text),
        chunk_size=chunk_size,
        overlap_units=overlap_sentences,
    )


def _format_slide_chunk(title: str, body: str) -> str:
    parts = []
    if title.strip():
        parts.append(f"[제목] {title.strip()}")
    if body.strip():
        parts.append(body.strip())
    return "\n".join(parts)


def _first_line_title_and_body(doc: dict[str, Any]) -> tuple[str, str]:
    """Infer a title from flat slide text without changing PPT preprocessing."""
    lines = _non_empty_lines(str(doc.get("text", "")))
    if not lines:
        return "", ""
    return lines[0], "\n".join(lines[1:])


def slide_aware_chunks(
    doc: dict[str, Any],
    chunk_size: int = SLIDE_AWARE_SIZE,
    overlap_sentences: int = SENTENCE_OVERLAP,
) -> list[dict[str, Any]]:
    """Chunk flat slide text without crossing a slide boundary.

    This strategy uses the original PPT text extraction.  The first non-empty
    line is repeated as a title and the remaining lines are packed without
    breaking line/sentence boundaries.
    """
    title, body = _first_line_title_and_body(doc)
    if not title:
        return []

    if not body:
        bodies = [""]
    elif len(body) <= chunk_size:
        bodies = [body]
    else:
        bodies = sentence_pack_chunk(
            body,
            chunk_size=chunk_size,
            overlap_sentences=overlap_sentences,
        )

    return [
        {
            "text": _format_slide_chunk(title, body_piece),
            "title": title,
            "group_piece_index": body_index,
            "element_type": "flat_slide_text",
        }
        for body_index, body_piece in enumerate(bodies)
    ]


def title_body_chunks(
    doc: dict[str, Any],
    chunk_size: int = SLIDE_AWARE_SIZE,
    overlap_sentences: int = SENTENCE_OVERLAP,
) -> list[dict[str, Any]]:
    """Create title-body chunks while preserving slide and line boundaries.

    All body groups on a slide are first assembled in visual order.  Only when
    the assembled body exceeds ``chunk_size`` is it split by line/sentence
    boundaries.  This avoids producing one tiny chunk per decorative text box.
    """
    title = str(doc.get("title", "")).strip()
    groups = list(doc.get("body_groups") or [])

    if not groups:
        body = str(doc.get("text", "")).strip()
        if title and body.startswith(title):
            body = body[len(title) :].lstrip()
        groups = [{"shape_id": None, "type": "text", "text": body, "lines": _non_empty_lines(body)}]

    body_groups = [str(group.get("text", "")).strip() for group in groups]
    body_groups = [body for body in body_groups if body]

    if not body_groups:
        if not title:
            return []
        return [
            {
                "text": _format_slide_chunk(title, ""),
                "title": title,
                "group_index": 0,
                "group_piece_index": 0,
                "shape_ids": [],
                "element_types": ["title"],
            }
        ]

    combined_body = "\n".join(body_groups)
    if len(combined_body) <= chunk_size:
        bodies = [combined_body]
    else:
        bodies = sentence_pack_chunk(
            combined_body,
            chunk_size=chunk_size,
            overlap_sentences=overlap_sentences,
        )

    shape_ids = [
        group.get("shape_id")
        for group in groups
        if group.get("shape_id") is not None
    ]
    element_types = sorted({str(group.get("type", "text")) for group in groups})

    pieces: list[dict[str, Any]] = []
    for body_index, body_piece in enumerate(bodies):
        pieces.append(
            {
                "text": _format_slide_chunk(title, body_piece),
                "title": title,
                "group_index": 0,
                "group_piece_index": body_index,
                "shape_ids": shape_ids,
                "element_types": element_types,
            }
        )
    return pieces


def chunk_documents(
    docs: list[dict[str, Any]],
    strategy: str = DEFAULT_STRATEGY,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
    document_id: str = "ajin_training_250416",
) -> list[dict[str, Any]]:
    """Convert extracted slides to a common chunk schema for all strategies."""
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}; choose one of {SUPPORTED_STRATEGIES}"
        )

    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        slide_no = int(doc["slide_no"])

        if strategy == "recursive":
            size = CHUNK_SIZE if chunk_size is None else chunk_size
            overlap_value = OVERLAP if overlap is None else overlap
            raw_pieces = [
                {"text": text}
                for text in recursive_chunk(doc.get("text", ""), size, overlap_value)
            ]
        elif strategy == "sentence_pack":
            size = SENTENCE_PACK_SIZE if chunk_size is None else chunk_size
            overlap_value = SENTENCE_OVERLAP if overlap is None else overlap
            raw_pieces = [
                {"text": text}
                for text in sentence_pack_chunk(
                    doc.get("text", ""),
                    chunk_size=size,
                    overlap_sentences=overlap_value,
                )
            ]
        elif strategy == "slide_aware":
            size = SLIDE_AWARE_SIZE if chunk_size is None else chunk_size
            overlap_value = SENTENCE_OVERLAP if overlap is None else overlap
            raw_pieces = slide_aware_chunks(
                doc,
                chunk_size=size,
                overlap_sentences=overlap_value,
            )
        else:
            size = SLIDE_AWARE_SIZE if chunk_size is None else chunk_size
            overlap_value = SENTENCE_OVERLAP if overlap is None else overlap
            raw_pieces = title_body_chunks(
                doc,
                chunk_size=size,
                overlap_sentences=overlap_value,
            )

        for slide_chunk_index, piece in enumerate(raw_pieces):
            text = piece["text"].strip()
            if not text:
                continue
            all_chunks.append(
                {
                    "chunk_id": (
                        f"{document_id}-p{slide_no:03d}-c{slide_chunk_index + 1:02d}"
                    ),
                    "slide_no": slide_no,
                    "slide_chunk_index": slide_chunk_index,
                    "text": text,
                    "title": piece.get("title", doc.get("title", "")),
                    "chunk_method": strategy,
                    "chunk_size": size,
                    "overlap": overlap_value,
                    **{
                        key: value
                        for key, value in piece.items()
                        if key not in {"text", "title"}
                    },
                }
            )
    return all_chunks


CHUNKING_CONFIGS = {
    "recursive": {"chunk_size": 150, "overlap": 30},
    "sentence_pack": {"chunk_size": 300, "overlap": 1},
    "slide_aware": {"chunk_size": 300, "overlap": 1},
    "title_body": {"chunk_size": 300, "overlap": 1},
}


def build_all_chunk_sets(
    docs: list[dict[str, Any]],
    structured_docs: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build four strategies, using structured slides only for title-body."""
    structured = docs if structured_docs is None else structured_docs
    return {
        strategy: chunk_documents(
            structured if strategy == "title_body" else docs,
            strategy=strategy,
            **config,
        )
        for strategy, config in CHUNKING_CONFIGS.items()
    }


if __name__ == "__main__":
    import argparse

    from documents import extract_slide_structures, extract_slide_texts

    parser = argparse.ArgumentParser(description="Compare PPT chunking strategies.")
    parser.add_argument("pptx", help="Path to the PPTX file")
    args = parser.parse_args()

    flat_documents = extract_slide_texts(args.pptx)
    structured_documents = extract_slide_structures(args.pptx)
    for method, chunks in build_all_chunk_sets(
        flat_documents,
        structured_documents,
    ).items():
        lengths = [len(chunk["text"]) for chunk in chunks]
        print(
            f"{method:14s} chunks={len(chunks):4d} "
            f"avg={sum(lengths) / len(lengths):6.1f} "
            f"min={min(lengths):4d} max={max(lengths):4d}"
        )

"""PowerPoint text extraction with title/body structure preserved."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER


TITLE_TYPES = {
    PP_PLACEHOLDER.TITLE,
    PP_PLACEHOLDER.CENTER_TITLE,
    PP_PLACEHOLDER.VERTICAL_TITLE,
}


def _leaf_shapes(shapes) -> Iterable:
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _leaf_shapes(shape.shapes)
        else:
            yield shape


def _text_lines(shape) -> list[str]:
    lines = []
    for paragraph in shape.text_frame.paragraphs:
        text = "".join(run.text for run in paragraph.runs).strip()
        if text:
            lines.append(text)
    return lines


def _table_lines(shape) -> list[str]:
    lines = []
    for row in shape.table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        if any(cells):
            lines.append(" | ".join(cells))
    return lines


def _is_title(shape) -> bool:
    if not getattr(shape, "is_placeholder", False):
        return False
    try:
        return shape.placeholder_format.type in TITLE_TYPES
    except (AttributeError, KeyError, ValueError):
        return False


def extract_slide_structures(pptx_path: str | Path) -> list[dict[str, Any]]:
    """Return slide number, title, ordered body groups, and flat text."""
    path = Path(pptx_path)
    if not path.exists():
        raise FileNotFoundError(f"PPTX 파일을 찾을 수 없습니다: {path}")

    presentation = Presentation(str(path))
    results = []
    for slide_no, slide in enumerate(presentation.slides, start=1):
        title = ""
        elements = []
        for shape in _leaf_shapes(slide.shapes):
            if getattr(shape, "has_table", False):
                lines = _table_lines(shape)
                kind = "table"
            elif getattr(shape, "has_text_frame", False):
                lines = _text_lines(shape)
                kind = "text"
            else:
                continue
            if not lines:
                continue
            text = "\n".join(lines)
            if kind == "text" and _is_title(shape):
                title = text
                continue
            elements.append(
                {
                    "shape_id": int(shape.shape_id),
                    "type": kind,
                    "text": text,
                    "lines": lines,
                    "top": int(getattr(shape, "top", 0)),
                    "left": int(getattr(shape, "left", 0)),
                }
            )

        elements.sort(key=lambda value: (value["top"], value["left"]))
        if not title:
            candidates = [
                value
                for value in elements
                if value["type"] == "text" and len(value["text"]) <= 120
            ]
            if candidates:
                title = candidates[0]["text"]
                elements.remove(candidates[0])

        flat = [value for value in [title, *[e["text"] for e in elements]] if value]
        results.append(
            {
                "slide_no": slide_no,
                "title": title,
                "body_groups": elements,
                "text": "\n".join(flat),
            }
        )
    return results


def extract_slide_texts(pptx_path: str | Path) -> list[dict[str, Any]]:
    """Return the backward-compatible flat slide schema."""
    return [
        {"slide_no": item["slide_no"], "text": item["text"]}
        for item in extract_slide_structures(pptx_path)
    ]


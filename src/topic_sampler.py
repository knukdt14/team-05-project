"""PPT 기반 퀴즈 주제 후보를 추출하고 샘플링하는 헬퍼."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from documents import extract_slide_texts


def _normalize_topic(value: str) -> str:
    cleaned = " ".join(value.split())
    cleaned = cleaned.replace("•", "").replace("●", "").replace("▪", "").strip()
    cleaned = cleaned.rstrip(":-）)")
    return cleaned.strip()


def _is_useful_topic(value: str) -> bool:
    cleaned = _normalize_topic(value)
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if lowered in {"목차", "내용", "소개", "결론", "요약", "index"}:
        return False
    if len(cleaned) > 80:
        return False
    return True


def extract_topics(pptx_path: str | Path) -> list[str]:
    """슬라이드 텍스트에서 주제 후보를 추출한다."""
    slide_docs = extract_slide_texts(pptx_path)
    topics: list[str] = []
    for item in slide_docs:
        text = item.get("text", "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for candidate in lines[:3]:
            if _is_useful_topic(candidate):
                topics.append(_normalize_topic(candidate))
    return list(dict.fromkeys(topics))


def apply_manual_groups(topics: Iterable[str]) -> list[str]:
    """수동 그룹핑 규칙을 적용해 후보 목록을 정리한다."""
    normalized: list[str] = []
    for topic in topics:
        cleaned = _normalize_topic(topic)
        if not cleaned:
            continue
        if cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def sample_topics(topics: Iterable[str], *, n: int = 5, seed: int | None = None) -> list[str]:
    """주제 후보를 결정적 방식으로 샘플링한다."""
    values = list(topics)
    if not values:
        return []
    if n <= 0:
        return []
    if len(values) <= n:
        return values

    rng = random.Random(seed)
    sampled = values[:]
    rng.shuffle(sampled)
    return sampled[:n]

"""
topic_sampler.py

PPT 슬라이드에서 퀴즈 주제(토픽)를 자동으로 뽑아내고, 그중 일부를
랜덤하게 골라 main.py의 QUIZ_TOPICS로 넘겨주는 모듈.

배경: 지금까지 main.py의 QUIZ_TOPICS는 사람이 직접 하드코딩한
["중간날림의 원인"] 한 개뿐이었음. "정답이 여러 개인 퀴즈가 나온다"는
문제를 폭넓게 재현/디버깅하려면 다양한 주제로 여러 개 생성해봐야 하므로,
PPT 슬라이드 제목을 자동으로 긁어서 매번 다른 조합으로 테스트할 수 있게 함.

추출 방식:
    documents.extract_slide_structures()가 반환하는 슬라이드별 body_groups
    (위치순 정렬된 텍스트 조각들) 중 상위 몇 개 안에서, "번호 제목"
    패턴(예: "1-1 자동차 구조 이해")이나 짧은 소제목 형태를 찾아 토픽으로 씀.
    슬라이드 상단의 부서명 배너(문서 전체에 반복되는 title 필드)는 토픽으로
    쓰기에 너무 일반적이라 제외하고, body_groups 쪽에서 실제 소제목을 찾음.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from documents import extract_slide_structures

_NUMBERED = re.compile(r"^\d+-\d+\s+(.+)")
_DIVIDER = re.compile(r"^\d+부\b")
_BLOCKLIST = {"CONTENTS", "목차", "개요"}
_SYMBOL_PREFIXES = ("■", "●", "□", "▶", "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧")


def _clean_candidate(text: str) -> str | None:
    text = text.split("\n")[0].strip()
    if not text or text.startswith(_SYMBOL_PREFIXES):
        return None
    if "|" in text:  # 표(table) 행이 이어붙여진 텍스트는 제목이 아님
        return None
    match = _NUMBERED.match(text)
    if match:
        text = match.group(1).strip()
    elif re.fullmatch(r"\d+", text) or _DIVIDER.match(text):
        return None
    if text in _BLOCKLIST:
        return None
    if not (3 <= len(text) <= 40):
        return None
    return text


def extract_topics(pptx_path: str | Path) -> list[dict[str, Any]]:
    """PPT 전체에서 (슬라이드 번호, 토픽) 쌍을 뽑되, 중복 토픽은 처음 등장한
    슬라이드 기준으로 한 번만 남긴다.

    슬라이드마다 상위 3개 요소 중 "제목처럼 보이는 첫 번째 것"을 그 슬라이드의
    제목으로 확정한다. 그 제목이 이미 다른 슬라이드에서 나온 적 있으면(중복
    섹션) 이 슬라이드는 통째로 건너뛴다 -- 다른 무관한 본문 요소로 넘어가서
    표 내용이나 캡션을 잘못 제목으로 주워오는 걸 방지하기 위함.
    부서명 배너(문서 전체에 반복되는 title)와 우연히 겹치는 본문 요소도 제외."""
    docs = extract_slide_structures(pptx_path)
    banners = {d["title"].strip() for d in docs if d["title"].strip()}
    seen: set[str] = set()
    topics: list[dict[str, Any]] = []
    for doc in docs:
        candidate = None
        for element in doc["body_groups"][:3]:
            candidate = _clean_candidate(element["text"])
            if candidate is not None and candidate in banners:
                candidate = None  # 부서 배너와 우연히 겹치면 무효 처리, 다음 요소 시도
                continue
            if candidate is not None:
                break  # 유효한 제목 후보를 찾았으면, 중복 여부와 무관하게 여기서 멈춤
        if candidate and candidate not in seen:
            seen.add(candidate)
            topics.append({"topic": candidate, "slide_no": doc["slide_no"]})
    return topics


def sample_topics(
    topics: list[dict[str, Any]],
    n: int,
    *,
    seed: int | None = None,
) -> list[str]:
    """추출된 토픽 중 n개를 무작위로 뽑아 문자열 리스트로 반환.
    seed를 지정하면 같은 조합이 재현되어 디버깅할 때 편리함."""
    rng = random.Random(seed)
    pool = topics.copy()
    n = min(n, len(pool))
    picked = rng.sample(pool, n)
    return [item["topic"] for item in picked]


# 자동 추출 결과 중, 글자는 다르지만 같은 상위 주제로 봐야 할 것들을
# 사람이 직접 확인하고 묶은 그룹. 여기 없는 토픽은 그대로 유지됨.
MANUAL_GROUPS: dict[str, list[str]] = {
    "차체(구조/제조공정)": [
        "제품현황 (차체)", "1. 자동차(차체) 구조 이해", "자동차(차체) 구조 이해",
        "차체 제조공정", "차체 제조공정_원자재", "차체 제조공정_프레스/금형",
        "차체 제조공정_조립", "차체 제조공정_품질관리", "자동차의 구조",
    ],
    "안전관리": ["1. 안전관리", "안전관리"],
    "주요업무": ["주요업무(1)", "주요업무(2)", "주요업무(3)"],
    "관련시스템": [f"관련시스템({i})" for i in range(1, 8)],
    "이슈별 관계법령": ["이슈별 관계법령(1)", "이슈별 관계법령(2)", "이슈별 관계법령(3)"],
    "JIG(구성/유틸리티/관리)": [
        "2. JIG의 구성 및 제작", "③ 유틸리티(UTILITY)", "3. JIG의 관리",
    ],
    "수소연료전지 원리": ["1-1.수소연료전지 원리", "1-2.수소연료전지 원리"],
    "공법계획(개념/해석/생산방식/PRO)": [
        "1) 공법이란?", "2) 해석이란?", "3) 생산방식", "4) PRO 이해",
    ],
}


def apply_manual_groups(
    topics: list[dict[str, Any]],
    groups: dict[str, list[str]] = MANUAL_GROUPS,
) -> list[dict[str, Any]]:
    """MANUAL_GROUPS에 정의된 대로 여러 토픽을 하나의 상위 토픽으로 합친다.
    같은 그룹에 속한 원래 슬라이드 번호는 리스트로 모아서 보존한다."""
    by_topic = {t["topic"]: t["slide_no"] for t in topics}
    grouped_members: set[str] = set()
    for members in groups.values():
        grouped_members.update(members)

    result: list[dict[str, Any]] = []
    for group_name, members in groups.items():
        slide_nos = [by_topic[m] for m in members if m in by_topic]
        if slide_nos:
            result.append({"topic": group_name, "slide_no": slide_nos})

    for t in topics:
        if t["topic"] not in grouped_members:
            result.append({"topic": t["topic"], "slide_no": [t["slide_no"]]})
    return result


if __name__ == "__main__":
    import sys

    pptx_path = sys.argv[1] if len(sys.argv) > 1 else "../data/경북대 교육 발표자료 250416-1.pptx"
    topics = extract_topics(pptx_path)
    print(f"추출된 고유 토픽: {len(topics)}개")

    grouped = apply_manual_groups(topics)
    print(f"수동 그룹핑 적용 후: {len(grouped)}개\n")

    sample = sample_topics(grouped, 5, seed=42)
    print("랜덤 샘플(seed=42, 재현 가능):")
    for t in sample:
        print(" -", t)

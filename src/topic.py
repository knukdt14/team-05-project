"""
src/topic.py

LLM(Solar)이 각 슬라이드를 읽고 퀴즈 주제를 '요약해서' 뽑아낸다.

기존 topic_sampler.py는 슬라이드 원문 줄을 그대로 긁어와서
"1.버니어 캘리퍼스 : 물체의 외경...0.05mm..." 같은 문장 조각이 주제가 되는 문제가 있었다.
이 모듈은 슬라이드 내용을 LLM에게 읽히고 "핵심 개념"을 명사구로 요약하게 해서
깔끔한 주제("버니어 캘리퍼스의 측정 원리")를 뽑는다.

- 생성용 LLM(quiz_generator.load_llm = Solar API)을 그대로 재사용.
- 슬라이드 수만큼 API 호출이라, 결과를 JSON에 캐시해 두 번째부터는 재호출하지 않는다.

실행:  cd src && python -m topic
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from documents import extract_slide_texts
from settings import PROJECT_ROOT
from quiz_generator import load_llm, _invoke, extract_json

DEFAULT_PPTX = PROJECT_ROOT / "data" / "경북대 교육 발표자료 250416-1.pptx"
DEFAULT_CACHE = PROJECT_ROOT / "results" / "llm_topics.json"

MIN_SLIDE_CHARS = 15   # 이보다 짧은 슬라이드(표지/빈칸)는 주제 추출 스킵
MAX_TOPIC_CHARS = 30   # 이보다 긴 결과는 요약 실패로 보고 버림


TOPIC_PROMPT = """\
너는 신입사원 교육자료에서 퀴즈 주제를 뽑는 도우미다.
아래 [슬라이드] 내용을 읽고, 신입사원 퀴즈로 낼 만한 핵심 주제를 뽑아라.

규칙:
- 주제는 한국어 명사구로 15자 이내로 짧게 작성한다.
- 원문 문장을 그대로 복사하지 말고 개념을 요약한다.
- 표지, 목차, 인사말, 회사 소개 등 퀴즈로 부적절하면 "없음"으로 답한다.
- 핵심 주제가 여러 개면 가장 중요한 것 1개만 고른다.
- 한국어로만 작성한다.

설명 없이 JSON 하나만 출력하라:
{{"topic":"핵심 주제 또는 없음"}}

[슬라이드 {slide_no}]
{text}
"""


def _clean_topic(topic: str) -> str:
    """앞의 번호·기호·공백을 제거해 깔끔한 명사구로 정리한다."""
    topic = " ".join(topic.split())
    topic = re.sub(r"^[\d①-⑳가-힣]{1,2}[\.\)]\s*", "", topic)  # "1." "3)" "가." 제거
    topic = re.sub(r"^[\-•·※\s]+", "", topic)                    # 앞 불릿/기호 제거
    return topic.strip()


def extract_topic_from_slide(generator, slide_no: int, text: str) -> str | None:
    """슬라이드 1개 -> LLM으로 주제 1개 (없거나 실패하면 None)."""
    if len(text.strip()) < MIN_SLIDE_CHARS:
        return None
    prompt = TOPIC_PROMPT.format(slide_no=slide_no, text=text[:1500])  # 너무 길면 자름
    raw = _invoke(generator, [{"role": "user", "content": prompt}], max_new_tokens=60)
    try:
        topic = _clean_topic(str(extract_json(raw).get("topic", "")))
    except (ValueError, json.JSONDecodeError):
        return None
    if not topic or topic == "없음" or len(topic) > MAX_TOPIC_CHARS:
        return None
    return topic


def extract_topics_llm(
    pptx_path=DEFAULT_PPTX,
    generator=None,
    *,
    cache_path=DEFAULT_CACHE,
    use_cache: bool = True,
) -> list[str]:
    """모든 슬라이드를 LLM으로 읽어 주제 목록을 만든다. 캐시가 있으면 재사용."""
    cache = Path(cache_path) if cache_path else None
    if use_cache and cache and cache.exists():
        print(f"캐시 사용: {cache}")
        return json.loads(cache.read_text(encoding="utf-8"))

    if generator is None:
        print("주제 추출용 LLM(Solar) 준비...")
        generator = load_llm()

    slides = extract_slide_texts(pptx_path)
    topics: list[str] = []
    for slide in slides:
        topic = extract_topic_from_slide(
            generator, slide["slide_no"], slide.get("text", "")
        )
        if topic and topic not in topics:
            topics.append(topic)
            print(f"슬라이드 {slide['slide_no']:3d} -> {topic}")

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(topics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n저장: {cache} (주제 {len(topics)}개)")
    return topics


def sample_topics(topics: list[str], *, n: int = 5, seed: int | None = None) -> list[str]:
    """주제 후보를 결정적(시드 고정) 방식으로 n개 샘플링한다."""
    values = list(topics)
    if not values or n <= 0:
        return []
    if len(values) <= n:
        return values
    rng = random.Random(seed)
    sampled = values[:]
    rng.shuffle(sampled)
    return sampled[:n]


if __name__ == "__main__":
    result = extract_topics_llm(use_cache=False)  # 처음엔 새로 뽑기
    print(f"\n{'=' * 50}")
    print(f"추출된 주제 {len(result)}개")
    print("=" * 50)

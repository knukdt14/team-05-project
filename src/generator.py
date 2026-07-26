"""Generate source-grounded quizzes with a fixed dictionary schema."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SUPPORTED_QUIZ_TYPES = {"multiple_choice", "short_answer", "true_false"}
REQUIRED_FIELDS = (
    "quiz_id",
    "type",
    "question",
    "choices",
    "answer",
    "explanation",
    "evidence",
    "source",
)

SYSTEM_PROMPT_RAG = """\
당신은 교육자료를 바탕으로 퀴즈를 만드는 출제자입니다.
반드시 [검색 자료]에 명시된 사실만 사용해 퀴즈 한 문제를 만드세요.
자료에 없는 내용을 추측하거나 외부 지식을 추가하지 마세요.

출력 규칙:
1. 설명이나 마크다운 없이 유효한 JSON 객체 하나만 출력합니다.
2. 키는 quiz_id, type, question, choices, answer, explanation, evidence, source를
   빠짐없이 포함합니다.
3. 객관식(type=multiple_choice)은 choices를 4개 만들고 answer는 정답 보기의
   1부터 시작하는 번호입니다.
4. OX(type=true_false)는 choices를 ["O", "X"]로 만들고 answer는 1 또는 2입니다.
5. 단답형(type=short_answer)은 choices를 빈 배열로 만들고 answer는 정답 문자열입니다.
6. evidence에는 어느 자료의 어떤 문장이 근거인지 자연어로 적습니다.
7. source에는 실제 사용한 한 청크의 file, slide, chunk_id를 그대로 적습니다.

[검색 자료]
{context}
"""

SYSTEM_PROMPT_BASELINE = """\
당신은 교육용 퀴즈 출제자입니다. 주어진 주제로 퀴즈 한 문제를 만드세요.
설명이나 마크다운 없이 quiz_id, type, question, choices, answer, explanation,
evidence, source 키가 있는 유효한 JSON 객체 하나만 출력하세요.
"""


def load_generator(model_name: str = MODEL_NAME):
    """Load the Hugging Face text-generation pipeline lazily."""
    from transformers import pipeline

    return pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
    )


def _generated_content(output: Any) -> str:
    """Extract assistant text from common transformers pipeline outputs."""
    if not isinstance(output, list) or not output:
        raise ValueError("모델 출력이 비어 있습니다.")

    generated = output[0]
    if isinstance(generated, dict):
        generated = generated.get("generated_text", generated)

    if isinstance(generated, list):
        for message in reversed(generated):
            if isinstance(message, dict) and "content" in message:
                return str(message["content"])
    if isinstance(generated, str):
        return generated

    raise ValueError("모델 출력에서 생성 텍스트를 찾을 수 없습니다.")


def parse_quiz_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object, tolerating code fences or leading text."""
    decoder = json.JSONDecoder()
    first_object: dict[str, Any] | None = None
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            if first_object is None:
                first_object = value
            if all(field in value for field in REQUIRED_FIELDS):
                return value
    if first_object is not None:
        return first_object
    raise ValueError(f"유효한 JSON 객체를 찾지 못했습니다: {text[:200]!r}")


def _normalize_chunks(
    retrieved_chunks: list[dict[str, Any] | str],
    source_file: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        if isinstance(chunk, str):
            normalized.append(
                {
                    "chunk_id": f"retrieved-c{index:02d}",
                    "slide_no": 0,
                    "text": chunk,
                    "file": source_file,
                }
            )
            continue

        normalized.append(
            {
                "chunk_id": str(chunk.get("chunk_id", f"retrieved-c{index:02d}")),
                "slide_no": int(chunk.get("slide_no", chunk.get("slide", 0))),
                "text": str(chunk.get("text", "")),
                "file": str(chunk.get("file", source_file)),
            }
        )
    if not normalized:
        raise ValueError("검색된 청크가 한 개 이상 필요합니다.")
    return normalized


def _context_text(chunks: list[dict[str, Any]]) -> str:
    sections = []
    for chunk in chunks:
        sections.append(
            "[자료: {file} | 슬라이드: {slide_no} | 청크 ID: {chunk_id}]\n{text}".format(
                **chunk
            )
        )
    return "\n\n---\n\n".join(sections)


def _validate_and_normalize(
    quiz: dict[str, Any],
    *,
    quiz_id: str,
    quiz_type: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in quiz]
    if missing:
        raise ValueError(f"퀴즈 출력에 필수 필드가 없습니다: {', '.join(missing)}")

    if quiz_type not in SUPPORTED_QUIZ_TYPES:
        raise ValueError(f"지원하지 않는 퀴즈 유형입니다: {quiz_type}")

    choices = quiz["choices"]
    answer = quiz["answer"]
    if quiz_type in {"multiple_choice", "true_false"} and isinstance(answer, str):
        if answer.strip().isdigit():
            answer = int(answer.strip())
    if not isinstance(choices, list):
        raise ValueError("choices는 리스트여야 합니다.")
    if quiz_type == "multiple_choice":
        if len(choices) != 4:
            raise ValueError("객관식 choices는 정확히 4개여야 합니다.")
        if not isinstance(answer, int) or isinstance(answer, bool) or not 1 <= answer <= 4:
            raise ValueError("객관식 answer는 1~4의 정수여야 합니다.")
    elif quiz_type == "true_false":
        if choices != ["O", "X"]:
            raise ValueError('OX 문제 choices는 ["O", "X"]여야 합니다.')
        if not isinstance(answer, int) or isinstance(answer, bool) or answer not in (1, 2):
            raise ValueError("OX 문제 answer는 1 또는 2여야 합니다.")
    else:
        if choices:
            raise ValueError("단답형 choices는 빈 리스트여야 합니다.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("단답형 answer는 비어 있지 않은 문자열이어야 합니다.")

    for field in ("question", "explanation", "evidence"):
        if not isinstance(quiz[field], str) or not quiz[field].strip():
            raise ValueError(f"{field}는 비어 있지 않은 문자열이어야 합니다.")

    source = quiz.get("source")
    requested_chunk_id = source.get("chunk_id") if isinstance(source, dict) else None
    selected = next(
        (chunk for chunk in chunks if chunk["chunk_id"] == requested_chunk_id),
        chunks[0],
    )

    # 식별자와 출처 메타데이터는 모델의 임의 생성을 허용하지 않고 실제 입력값으로 고정한다.
    return {
        "quiz_id": quiz_id,
        "type": quiz_type,
        "question": quiz["question"].strip(),
        "choices": choices,
        "answer": answer,
        "explanation": quiz["explanation"].strip(),
        "evidence": quiz["evidence"].strip(),
        "source": {
            "file": selected["file"],
            "slide": selected["slide_no"],
            "chunk_id": selected["chunk_id"],
        },
    }


def generate_rag(
    gen,
    topic: str,
    retrieved_chunks: list[dict[str, Any] | str],
    *,
    quiz_id: str = "quiz-001",
    quiz_type: str = "multiple_choice",
    source_file: str = "자료 1",
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    """Generate one grounded quiz and return a validated Python dictionary."""
    chunks = _normalize_chunks(retrieved_chunks, source_file)
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_RAG.format(context=_context_text(chunks)),
        },
        {
            "role": "user",
            "content": (
                f"주제: {topic}\n"
                f"quiz_id: {quiz_id}\n"
                f"type: {quiz_type}\n"
                "위 조건에 맞는 문제 한 개를 JSON으로 생성하세요."
            ),
        },
    ]
    output = gen(messages, max_new_tokens=max_new_tokens)
    quiz = parse_quiz_json(_generated_content(output))
    return _validate_and_normalize(
        quiz,
        quiz_id=quiz_id,
        quiz_type=quiz_type,
        chunks=chunks,
    )


def generate_baseline(
    gen,
    topic: str,
    *,
    quiz_id: str = "quiz-001",
    quiz_type: str = "multiple_choice",
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    """Generate an ungrounded comparison quiz using the same output schema."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_BASELINE},
        {
            "role": "user",
            "content": (
                f"주제: {topic}\nquiz_id: {quiz_id}\ntype: {quiz_type}\n"
                'source는 {"file": "baseline", "slide": 0, '
                '"chunk_id": "baseline"}으로 작성하세요.'
            ),
        },
    ]
    output = gen(messages, max_new_tokens=max_new_tokens)
    quiz = parse_quiz_json(_generated_content(output))
    return _validate_and_normalize(
        quiz,
        quiz_id=quiz_id,
        quiz_type=quiz_type,
        chunks=[
            {
                "file": "baseline",
                "slide_no": 0,
                "chunk_id": "baseline",
                "text": "",
            }
        ],
    )


def format_quiz(quiz: dict[str, Any]) -> str:
    """Return readable JSON while retaining the dictionary as the main API."""
    return json.dumps(quiz, ensure_ascii=False, indent=2)


def save_quiz_result(
    quiz: dict[str, Any],
    output_dir: str | Path | None = None,
    filename: str | None = None,
) -> Path:
    """Validate basic schema and save a quiz as UTF-8 JSON under ``result``."""
    missing = [field for field in REQUIRED_FIELDS if field not in quiz]
    if missing:
        raise ValueError(f"퀴즈 출력에 필수 필드가 없습니다: {', '.join(missing)}")

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "result"
    result_dir = Path(output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    safe_quiz_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(quiz["quiz_id"]))
    output_name = filename or f"{safe_quiz_id}.json"
    if not output_name.lower().endswith(".json"):
        output_name += ".json"

    output_path = result_dir / output_name
    output_path.write_text(
        json.dumps(quiz, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


if __name__ == "__main__":
    print("generator.py는 import해서 사용하세요.")
    print("generate_rag(...)의 반환형은 dict입니다.")

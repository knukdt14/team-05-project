"""Qwen-3B quiz generation with schema checks and single-answer validation."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from transformers import AutoTokenizer, GenerationConfig, pipeline


FINAL_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"


class QuizDraft(BaseModel):
    type: Literal["multiple_choice", "true_false", "short_answer"]
    question: str = Field(min_length=1)
    choices: list[str]
    answer: int | str
    explanation: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    source_ref: int = Field(ge=1)


SYSTEM_PROMPT = """\
너는 신입사원 교육용 퀴즈 출제자다.
아래 [참고자료]에 명시된 내용만 사용하고 외부 지식을 추가하지 마라.

객관식 생성 규칙:
- 보기는 정확히 4개다.
- 정답은 정확히 하나만 존재해야 한다.
- answer는 정답 보기의 1부터 시작하는 번호다.
- 오답 3개는 참고자료에서 정답으로 확인되지 않아야 한다.
- 자료에서 같은 분류로 나열된 여러 정답을 정답과 오답으로 섞지 마라.
- 단일 정답 문제를 만들 수 없으면 질문의 범위를 좁혀라.

OX 문제는 choices=["O","X"], answer는 1 또는 2로 작성한다.
단답형은 choices=[], answer는 정답 문자열로 작성한다.
evidence는 참고자료의 근거 문장을 작성한다.
source_ref는 사용한 자료 번호다.

설명이나 코드 블록 없이 다음 키를 가진 JSON 객체 하나만 출력하라:
type, question, choices, answer, explanation, evidence, source_ref

[참고자료]
{context}
"""


JUDGE_PROMPT = """\
아래 참고자료만 기준으로 객관식 보기 각각이 질문의 정답인지 판정하라.
정답 보기 번호는 1부터 시작한다. 설명 없이 JSON 하나만 출력하라.

출력:
{{"correct_choice_indices":[1],"ambiguous_choice_indices":[],"reason":"판정 이유"}}

[참고자료]
{context}

[퀴즈]
{quiz}
"""


def load_llm(model_name: str = FINAL_LLM_MODEL):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 사용할 수 없습니다. GPU용 PyTorch 환경에서 실행하세요."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        clean_up_tokenization_spaces=False,
    )
    return pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        device=0,
        torch_dtype=torch.float16,
    )


def _content(output: Any) -> str:
    generated = output[0].get("generated_text", output[0])
    if isinstance(generated, list):
        for message in reversed(generated):
            if isinstance(message, dict) and "content" in message:
                return str(message["content"])
    return str(generated)


def _invoke(generator, messages: list[dict], max_new_tokens: int = 512) -> str:
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        max_length=None,
        do_sample=False,
        pad_token_id=getattr(getattr(generator, "tokenizer", None), "eos_token_id", None),
    )
    output = generator(
        messages,
        generation_config=generation_config,
        return_full_text=False,
    )
    return _content(output)


def extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("모델 출력에서 JSON 객체를 찾지 못했습니다.")


def build_context(retrieved: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        (
            f"자료 {index} | 슬라이드 {item['slide_no']} | "
            f"청크 {item['chunk_id']}\n{item['text']}"
        )
        for index, item in enumerate(retrieved, start=1)
    )


def _validate_structure(draft: QuizDraft, requested_type: str) -> None:
    if draft.type != requested_type:
        raise ValueError(f"요청 유형은 {requested_type}, 생성 유형은 {draft.type}입니다.")
    if draft.type == "multiple_choice":
        if len(draft.choices) != 4 or len(set(draft.choices)) != 4:
            raise ValueError("객관식 보기는 중복 없이 정확히 4개여야 합니다.")
        if (
            not isinstance(draft.answer, int)
            or isinstance(draft.answer, bool)
            or not 1 <= draft.answer <= 4
        ):
            raise ValueError("객관식 answer는 1~4의 정수여야 합니다.")
    elif draft.type == "true_false":
        if (
            draft.choices != ["O", "X"]
            or not isinstance(draft.answer, int)
            or isinstance(draft.answer, bool)
            or draft.answer not in (1, 2)
        ):
            raise ValueError('OX는 choices=["O","X"], answer=1 또는 2여야 합니다.')
    else:
        if draft.choices or not isinstance(draft.answer, str) or not draft.answer.strip():
            raise ValueError("단답형 choices는 빈 배열이고 answer는 문자열이어야 합니다.")


def _single_answer_check(generator, context: str, draft: QuizDraft) -> tuple[bool, str]:
    prompt = JUDGE_PROMPT.format(
        context=context,
        quiz=json.dumps(draft.model_dump(), ensure_ascii=False),
    )
    raw = _invoke(
        generator,
        [{"role": "user", "content": prompt}],
        max_new_tokens=180,
    )
    result = extract_json(raw)
    correct = result.get("correct_choice_indices")
    ambiguous = result.get("ambiguous_choice_indices")
    expected = [int(draft.answer)]
    ok = correct == expected and ambiguous == []
    return ok, str(result.get("reason", "보기 검증 실패"))


def generate_quiz(
    generator,
    topic: str,
    retrieved: list[dict[str, Any]],
    *,
    quiz_id: str,
    quiz_type: str = "multiple_choice",
    file_label: str = "자료 1",
    max_attempts: int = 3,
    validate_choices: bool = True,
) -> dict[str, Any]:
    if not retrieved:
        raise ValueError("검색 결과가 없어 퀴즈를 생성할 수 없습니다.")

    context = build_context(retrieved)
    failure = ""
    for attempt in range(1, max_attempts + 1):
        user = f"주제: {topic}\n퀴즈 유형: {quiz_type}"
        if failure:
            user += f"\n이전 실패 이유: {failure}\n문제를 새로 만들어라."
        raw = _invoke(
            generator,
            [
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": user},
            ],
        )
        try:
            draft = QuizDraft.model_validate(extract_json(raw))
            _validate_structure(draft, quiz_type)
            if quiz_type == "multiple_choice" and validate_choices:
                valid, reason = _single_answer_check(generator, context, draft)
                if not valid:
                    raise ValueError(f"복수정답/모호성 검증 실패: {reason}")

            if draft.source_ref > len(retrieved):
                raise ValueError(
                    f"source_ref는 1~{len(retrieved)} 범위여야 합니다."
                )
            source_index = draft.source_ref - 1
            source_chunk = retrieved[source_index]
            return {
                "quiz_id": quiz_id,
                "type": draft.type,
                "question": draft.question.strip(),
                "choices": [choice.strip() for choice in draft.choices],
                "answer": draft.answer,
                "explanation": draft.explanation.strip(),
                "evidence": draft.evidence.strip(),
                "source": {
                    "file": file_label,
                    "slide": source_chunk["slide_no"],
                    "chunk_id": source_chunk["chunk_id"],
                },
            }
        except (ValueError, ValidationError, json.JSONDecodeError) as error:
            failure = str(error)

    raise RuntimeError(
        f"{max_attempts}회 생성 후에도 유효한 퀴즈를 만들지 못했습니다: {failure}"
    )

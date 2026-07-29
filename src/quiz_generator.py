"""Solar API 기반 퀴즈 생성 (스키마 검증 + 단일정답/부정형 설계)."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

import requests
from pydantic import BaseModel, Field, ValidationError


# 업스테이지 Solar API (OpenAI 호환). 경량/저렴은 "solar-mini".
FINAL_LLM_MODEL = "solar-pro"
UPSTAGE_ENDPOINT = "https://api.upstage.ai/v1/chat/completions"


class QuizDraft(BaseModel):
    type: Literal["multiple_choice", "true_false", "short_answer"]
    question: str = Field(min_length=1)
    choices: list[str]
    answer: int | str
    explanation: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    source_refs: list[int] = Field(min_length=1)


class MultipleChoiceTarget(BaseModel):
    question: str = Field(min_length=1)
    correct_answer: str = Field(min_length=1)
    other_valid_answers: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    source_refs: list[int] = Field(min_length=1)


class DistractorDraft(BaseModel):
    distractors: list[str]


class QuestionRewrite(BaseModel):
    question: str = Field(min_length=1)


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
source_refs는 근거로 사용한 자료 번호들의 리스트다. 예: [1] 또는 [1, 2]. 자료 번호(정수)만 넣어라(chunk_id 금지).

설명이나 코드 블록 없이 다음 키를 가진 JSON 객체 하나만 출력하라:
type, question, choices, answer, explanation, evidence, source_refs

[참고자료]
{context}
"""


TARGET_PROMPT = """\
너는 신입사원 교육용 객관식 퀴즈의 정답 설계자다.
아래 참고자료만 사용하여 주제에 관한 단일 정답 질문을 설계하라.

규칙:
- 먼저 주제에 답이 될 수 있는 사실이 여러 개인지 확인하라.
- 그중 하나의 원자적 사실을 correct_answer로 선택하라.
- correct_answer 하나만 답이 되도록 질문의 대상, 조건, 관계를 구체적으로 좁혀라.
- "원인은 무엇인가?"처럼 여러 항목이 답이 될 수 있는 질문을 만들지 마라.
- 원래 주제 또는 유사한 질문에 답이 될 수 있는 나머지 사실은 other_valid_answers에 모두 넣어라.
- correct_answer를 other_valid_answers에 넣지 마라.
- 질문에 correct_answer 단어(또는 그 일부·유사어·영문명)를 절대 포함하지 마라. 정답 명칭을 질문에 노출하면 안 된다.
- 정답의 명칭 대신 그 특징·용도·조건만 제시하고, 명칭은 보기에서 고르게 하라.
- 자료 문장을 그대로 옮겨 "이것의 이름은?" 식으로 정의를 낭독하지 마라. 구분·비교·적용을 묻는 형태로 만들어라.
- evidence는 참고자료의 실제 근거 문장이어야 한다.
- source_refs에는 사용한 자료 번호만 정수로 넣어라.
- null과 빈 문자열을 사용하지 마라.

JSON 객체 하나만 출력하라:
{{"question":"단일 정답 질문","correct_answer":"정답",
"other_valid_answers":["다른 유효 정답"],
"explanation":"설명","evidence":"근거 문장","source_refs":[1]}}

[주제]
{topic}

[참고자료]
{context}
"""


DISTRACTOR_PROMPT = """\
너는 객관식 퀴즈의 오답 설계자다.
아래 질문과 정답을 바꾸지 말고 새로운 오답 후보 {candidate_count}개를 생성하라.

규칙:
- distractors는 서로 다른 오답 후보 {candidate_count}개다.
- 정답 및 오답 금지 목록의 항목을 그대로 또는 바꿔 말해 사용하지 마라.
- 이미 채택한 오답과 이전에 거절된 오답을 다시 사용하지 마라.
- 참고자료 전체에서 질문의 정답으로 확인되는 내용을 오답으로 사용하지 마라.
- 핵심 조건, 대상, 방향 또는 상태를 바꿔 그럴듯하지만 명백히 틀린 오답을 만들어라.
- 단어 끝에 "불량", "부족" 등을 기계적으로 붙인 부자연스러운 표현을 만들지 마라.
- 같은 단어 또는 접미사를 반복하지 마라.
- 질문 범위를 벗어난 표현과 애매한 표현을 사용하지 마라.
- 설명 없이 JSON 객체 하나만 출력하라.

출력:
{{"distractors":["오답1","오답2","오답3"]}}

[질문]
{question}

[정답]
{correct_answer}

[오답 금지 목록]
{forbidden_answers}

[이미 채택한 오답]
{accepted_distractors}

[이전에 거절된 오답]
{rejected_distractors}

[참고자료]
{context}

[이전 검증 실패]
{failure}

[이번 생성 전략]
{strategy}
"""


TARGET_JUDGE_PROMPT = """\
아래 참고자료만 기준으로 질문이 단일 정답형인지 판정하라.

판정 규칙:
- correct_answer만 질문에 답할 수 있으면 single_answer=true다.
- other_valid_answers 중 하나라도 질문에 자연스럽게 답할 수 있으면 false다.
- "원인은 무엇인가?", "종류는 무엇인가?"처럼 여러 항목을 허용하는 질문은 false다.
- 설명 없이 JSON 객체 하나만 출력하라.

출력:
{{"single_answer":true,"reason":"판정 이유"}}

[질문]
{question}

[정답]
{correct_answer}

[다른 유효 정답]
{other_valid_answers}

[참고자료]
{context}
"""


QUESTION_REWRITE_PROMPT = """\
아래 넓은 질문을 correct_answer 하나만 답이 되는 구체적인 질문으로 다시 작성하라.
정답, 근거, 참고자료의 의미를 바꾸지 마라.

규칙:
- "원인은 무엇인가?", "종류는 무엇인가?"처럼 여러 답을 허용하는 표현을 금지한다.
- correct_answer의 대상, 조건, 관계 또는 상태를 evidence에서 찾아 질문에 명시하라.
- other_valid_answers가 자연스럽게 답할 수 없는 질문이어야 한다.
- 질문에 correct_answer 단어(또는 그 일부·유사어·영문명)를 절대 포함하지 마라.
- 정답 명칭 대신 특징·용도·조건만으로 추론하게 하고, 정의를 그대로 낭독하지 마라.
- JSON 객체 하나만 출력하라.

출력:
{{"question":"범위가 좁혀진 질문"}}

[기존 질문]
{question}

[정답]
{correct_answer}

[다른 유효 정답]
{other_valid_answers}

[근거]
{evidence}

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
    """Solar API 클라이언트를 준비한다(모델 다운로드·GPU 불필요).
    UPSTAGE_API_KEY 환경변수(.env 포함)가 필요하다."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("UPSTAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "UPSTAGE_API_KEY가 없습니다. .env 또는 환경변수에 키를 설정하세요."
        )
    # config가 Qwen 경로 등을 넘겨도 Solar 모델로 대체
    if "/" in model_name or "qwen" in model_name.lower():
        model_name = FINAL_LLM_MODEL
    return {"api_key": api_key, "model": model_name}


def _invoke(
    generator,
    messages: list[dict],
    max_new_tokens: int = 512,
    *,
    do_sample: bool = True,
) -> str:
    """Solar API(OpenAI 호환) 채팅 완성 호출."""
    payload = {
        "model": generator["model"],
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.7 if do_sample else 0.0,
    }
    response = requests.post(
        UPSTAGE_ENDPOINT,
        headers={
            "Authorization": f"Bearer {generator['api_key']}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


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
            f"자료 {index} | 슬라이드 {item['slide_no']}\n{item['text']}"
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


def _single_answer_assessment(
    generator,
    context: str,
    draft: QuizDraft,
) -> dict[str, Any]:
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
    return result


def _single_answer_check(generator, context: str, draft: QuizDraft) -> tuple[bool, str]:
    result = _single_answer_assessment(generator, context, draft)
    correct = result.get("correct_choice_indices")
    ambiguous = result.get("ambiguous_choice_indices")
    expected = [int(draft.answer)]
    ok = correct == expected and ambiguous == []
    return ok, str(result.get("reason", "보기 검증 실패"))


def _normalized_choice(value: str) -> str:
    return " ".join(value.split()).casefold()


def _validate_source_refs(source_refs: list[int], retrieved_count: int) -> None:
    for ref in source_refs:
        if not 1 <= ref <= retrieved_count:
            raise ValueError(
                f"source_refs 번호는 1~{retrieved_count} 범위여야 합니다: {ref}"
            )


def _target_single_answer_check(
    generator,
    context: str,
    target: MultipleChoiceTarget,
) -> tuple[bool, str]:
    prompt = TARGET_JUDGE_PROMPT.format(
        question=target.question,
        correct_answer=target.correct_answer,
        other_valid_answers=(
            "\n".join(f"- {answer}" for answer in target.other_valid_answers)
            or "없음"
        ),
        context=context,
    )
    raw = _invoke(
        generator,
        [{"role": "user", "content": prompt}],
        max_new_tokens=180,
    )
    result = extract_json(raw)
    return (
        result.get("single_answer") is True,
        str(result.get("reason", "질문 단일 정답성 검증 실패")),
    )


def _rewrite_target_question(
    generator,
    context: str,
    target: MultipleChoiceTarget,
) -> str:
    prompt = QUESTION_REWRITE_PROMPT.format(
        question=target.question,
        correct_answer=target.correct_answer,
        other_valid_answers=(
            "\n".join(f"- {answer}" for answer in target.other_valid_answers)
            or "없음"
        ),
        evidence=target.evidence,
        context=context,
    )
    raw = _invoke(
        generator,
        [{"role": "user", "content": prompt}],
        max_new_tokens=180,
    )
    return QuestionRewrite.model_validate(extract_json(raw)).question.strip()


def _generate_multiple_choice_target(
    generator,
    topic: str,
    context: str,
    *,
    retrieved_count: int,
    max_attempts: int,
) -> MultipleChoiceTarget:
    failure = ""
    for _ in range(max_attempts):
        prompt = TARGET_PROMPT.format(topic=topic, context=context)
        if failure:
            prompt += (
                "\n\n이전 출력의 실패 이유:\n"
                f"{failure}\n규칙을 지켜 완전히 새 JSON 객체를 작성하라."
            )
        raw = _invoke(generator, [{"role": "user", "content": prompt}])
        try:
            target = MultipleChoiceTarget.model_validate(extract_json(raw))
            _validate_source_refs(target.source_refs, retrieved_count)

            correct = _normalized_choice(target.correct_answer)
            seen = {correct}
            cleaned_other_answers = []
            for answer in target.other_valid_answers:
                normalized = _normalized_choice(answer)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                cleaned_other_answers.append(answer.strip())
            target.other_valid_answers = cleaned_other_answers

            valid, valid_reason = _target_single_answer_check(generator, context, target)
            if not valid:
                rewritten_question = _rewrite_target_question(
                    generator,
                    context,
                    target,
                )
                rewritten_target = target.model_copy(
                    update={"question": rewritten_question}
                )
                rewritten_valid, rewritten_reason = _target_single_answer_check(
                    generator,
                    context,
                    rewritten_target,
                )
                if not rewritten_valid:
                    # 정형(단일정답)으로 불가능한 목록형 주제.
                    # 억지 템플릿으로 때우지 않고 실패시켜, 상위에서 부정형으로 전환한다.
                    raise ValueError(
                        f"단일 정답 질문 설계 실패: {rewritten_reason or valid_reason}"
                    )
                target = rewritten_target

            # 질문에 정답 단어가 그대로 노출되면 재생성 (답이 뻔한 문제 방지)
            question_key = _normalized_choice(target.question).replace(" ", "")
            answer_key = _normalized_choice(target.correct_answer).replace(" ", "")
            if answer_key and answer_key in question_key:
                raise ValueError(
                    f"질문에 정답('{target.correct_answer}')이 노출됨 -> 재생성"
                )
            return target
        except (ValueError, ValidationError, json.JSONDecodeError) as error:
            failure = str(error)

    raise RuntimeError(
        f"{max_attempts}회 시도 후 단일 정답 질문을 설계하지 못했습니다: {failure}"
    )


def _assemble_multiple_choice(
    target: MultipleChoiceTarget,
    distractors: list[str],
    topic: str,
) -> QuizDraft:
    answer_position = sum(ord(character) for character in topic) % 4
    choices = list(distractors)
    choices.insert(answer_position, target.correct_answer)
    return QuizDraft(
        type="multiple_choice",
        question=target.question,
        choices=choices,
        answer=answer_position + 1,
        explanation=target.explanation,
        evidence=target.evidence,
        source_refs=target.source_refs,
    )


def _generate_validated_multiple_choice(
    generator,
    topic: str,
    context: str,
    *,
    retrieved_count: int,
    max_attempts: int,
    validate_choices: bool,
) -> QuizDraft:
    target = _generate_multiple_choice_target(
        generator,
        topic,
        context,
        retrieved_count=retrieved_count,
        max_attempts=max_attempts,
    )
    forbidden = [target.correct_answer, *target.other_valid_answers]
    forbidden_text = "\n".join(f"- {answer}" for answer in forbidden)
    forbidden_normalized = {
        _normalized_choice(answer)
        for answer in forbidden
        if answer.strip()
    }
    accepted: dict[str, str] = {}
    rejected: dict[str, str] = {}
    failure = "없음"
    strategies = (
        "정답의 핵심 상태를 반대 또는 정상 상태로 바꿔라.",
        "정답과 같은 문법 형태를 유지하되 대상이나 조건을 바꿔라.",
        "참고자료에 없는 원인-결과 관계를 사용하되 자연스러운 표현으로 작성하라.",
        "정답과 길이가 비슷한 간결한 명사구를 사용하라.",
        "앞선 후보와 전혀 다른 핵심 단어를 사용하라.",
    )

    for attempt in range(max_attempts):
        candidate_count = max(4, (3 - len(accepted)) * 2)
        prompt = DISTRACTOR_PROMPT.format(
            question=target.question,
            correct_answer=target.correct_answer,
            candidate_count=candidate_count,
            forbidden_answers=forbidden_text,
            accepted_distractors=(
                "\n".join(f"- {value}" for value in accepted.values()) or "없음"
            ),
            rejected_distractors=(
                "\n".join(f"- {value}" for value in rejected.values()) or "없음"
            ),
            context=context,
            failure=failure,
            strategy=strategies[attempt % len(strategies)],
        )
        raw = _invoke(
            generator,
            [{"role": "user", "content": prompt}],
            do_sample=True,
        )
        try:
            distractor_draft = DistractorDraft.model_validate(extract_json(raw))
            for candidate in distractor_draft.distractors:
                text = candidate.strip()
                normalized = _normalized_choice(text)
                if (
                    not normalized
                    or normalized in forbidden_normalized
                    or normalized in accepted
                    or normalized in rejected
                ):
                    continue
                accepted[normalized] = text

            if len(accepted) < 3:
                failure = (
                    "정답·다른 유효 정답·중복 보기를 제거한 뒤 "
                    f"사용 가능한 오답이 {len(accepted)}개뿐입니다. "
                    f"새로운 오답 {3 - len(accepted)}개 이상을 작성하라."
                )
                continue

            distractors = list(accepted.values())[:3]
            quiz = _assemble_multiple_choice(target, distractors, topic)
            _validate_structure(quiz, "multiple_choice")
            if not validate_choices:
                return quiz

            assessment = _single_answer_assessment(generator, context, quiz)
            correct_indices = assessment.get("correct_choice_indices") or []
            ambiguous_indices = assessment.get("ambiguous_choice_indices") or []
            expected = int(quiz.answer)
            if correct_indices == [expected] and ambiguous_indices == []:
                return quiz

            invalid_indices = {
                index
                for index in [*correct_indices, *ambiguous_indices]
                if isinstance(index, int) and index != expected
            }
            removed = []
            for index in invalid_indices:
                if not 1 <= index <= len(quiz.choices):
                    continue
                choice = quiz.choices[index - 1]
                normalized = _normalized_choice(choice)
                if normalized == _normalized_choice(target.correct_answer):
                    continue
                accepted.pop(normalized, None)
                rejected[normalized] = choice
                removed.append(choice)

            if not removed:
                for choice in distractors:
                    normalized = _normalized_choice(choice)
                    accepted.pop(normalized, None)
                    rejected[normalized] = choice

            reason = str(assessment.get("reason", "보기 검증 실패"))
            failure = (
                f"복수정답/모호성 검증 실패: {reason}. "
                "거절된 보기를 제외하고 새로운 오답 후보를 작성하라."
            )
        except (ValueError, ValidationError, json.JSONDecodeError) as error:
            failure = str(error)

    raise RuntimeError(
        f"{max_attempts}회 오답 재생성 후에도 단일 정답 보기를 만들지 못했습니다: "
        f"{failure}"
    )


def _quiz_result(
    draft: QuizDraft,
    retrieved: list[dict[str, Any]],
    *,
    quiz_id: str,
    file_label: str,
) -> dict[str, Any]:
    _validate_source_refs(draft.source_refs, len(retrieved))
    sources = []
    for ref in draft.source_refs:
        chunk = retrieved[ref - 1]
        sources.append(
            {
                "file": file_label,
                "slide": chunk["slide_no"],
                "chunk_id": chunk["chunk_id"],
            }
        )
    return {
        "quiz_id": quiz_id,
        "type": draft.type,
        "question": draft.question.strip(),
        "choices": [choice.strip() for choice in draft.choices],
        "answer": draft.answer,
        "explanation": draft.explanation.strip(),
        "evidence": draft.evidence.strip(),
        "sources": sources,
    }


class NegativeDraft(BaseModel):
    real_items: list[str] = Field(min_length=3)
    fake_item: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    source_refs: list[int] = Field(min_length=1)


NEGATIVE_PROMPT = """\
너는 신입사원 교육용 객관식 '부정형' 퀴즈 설계자다.
아래 참고자료만 사용하여 '{topic}'에 대한 부정형 문제를 설계하라.

부정형 문제는 "다음 중 ~에 해당하지 않는 것은?" 형태다.
- real_items: 참고자료에서 '{topic}'에 실제로 해당하는 서로 다른 항목 3개.
- fake_item: '{topic}'과 같은 분야처럼 보이지만 참고자료에는 없는 그럴듯한 가짜 1개.
- fake_item은 real_items와 길이·형식이 비슷해야 하고, 참고자료에서 정답으로 확인되면 안 된다.
- 모두 한국어로만 작성하고 null·빈 문자열을 쓰지 마라.
- source_refs에는 사용한 자료 번호만 정수로 넣어라.

설명 없이 JSON 하나만 출력하라:
{{"real_items":["항목1","항목2","항목3"],"fake_item":"가짜 항목",
"explanation":"fake_item이 왜 해당하지 않는지 설명","evidence":"근거 문장","source_refs":[1]}}

[주제]
{topic}

[참고자료]
{context}
"""


def _generate_negative_multiple_choice(
    generator,
    topic: str,
    context: str,
    *,
    retrieved_count: int,
    max_attempts: int,
) -> QuizDraft:
    """정형(단일정답) 설계가 불가능한 목록형 주제를 '부정형'으로 만든다.
    참고자료의 실제 항목 3개 + 자료 밖 가짜 1개 -> 정답은 가짜(해당 없음)."""
    failure = ""
    for _ in range(max_attempts):
        prompt = NEGATIVE_PROMPT.format(topic=topic, context=context)
        if failure:
            prompt += f"\n\n이전 실패 이유:\n{failure}\n새 JSON 객체를 작성하라."
        raw = _invoke(generator, [{"role": "user", "content": prompt}], do_sample=True)
        try:
            neg = NegativeDraft.model_validate(extract_json(raw))
            _validate_source_refs(neg.source_refs, retrieved_count)

            # 실제 항목 중복 제거
            reals: list[str] = []
            seen: set[str] = set()
            for item in neg.real_items:
                normalized = _normalized_choice(item)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    reals.append(item.strip())
            if len(reals) < 3:
                failure = "참고자료의 실제 항목(real_items)이 3개 미만입니다."
                continue

            fake = neg.fake_item.strip()
            if not fake or _normalized_choice(fake) in seen:
                failure = "fake_item이 비었거나 실제 항목과 겹칩니다."
                continue

            reals = reals[:3]
            position = sum(ord(character) for character in topic) % 4
            choices = list(reals)
            choices.insert(position, fake)   # 가짜(정답)를 결정적 위치에 삽입

            draft = QuizDraft(
                type="multiple_choice",
                question=f"다음 중 '{topic}'에 해당하지 않는 것은?",
                choices=choices,
                answer=position + 1,
                explanation=neg.explanation.strip(),
                evidence=neg.evidence.strip(),
                source_refs=neg.source_refs,
            )
            _validate_structure(draft, "multiple_choice")
            return draft
        except (ValueError, ValidationError, json.JSONDecodeError) as error:
            failure = str(error)

    raise RuntimeError(f"{max_attempts}회 시도 후 부정형 문제를 만들지 못했습니다: {failure}")


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
    if quiz_type == "multiple_choice":
        try:
            draft = _generate_validated_multiple_choice(
                generator,
                topic,
                context,
                retrieved_count=len(retrieved),
                max_attempts=max_attempts,
                validate_choices=validate_choices,
            )
        except RuntimeError:
            # 정형(단일정답)으로 안 되는 목록형 주제 -> 부정형("~아닌 것은?")으로 전환
            draft = _generate_negative_multiple_choice(
                generator,
                topic,
                context,
                retrieved_count=len(retrieved),
                max_attempts=max_attempts,
            )
        return _quiz_result(
            draft,
            retrieved,
            quiz_id=quiz_id,
            file_label=file_label,
        )

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
            return _quiz_result(
                draft,
                retrieved,
                quiz_id=quiz_id,
                file_label=file_label,
            )
        except (ValueError, ValidationError, json.JSONDecodeError) as error:
            failure = str(error)

    raise RuntimeError(
        f"{max_attempts}회 생성 후에도 유효한 퀴즈를 만들지 못했습니다: {failure}"
    )

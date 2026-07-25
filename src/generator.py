"""
src/generator.py  [C 담당: 로컬 LLM 퀴즈 생성]

검색된 청크(retrieved chunks)를 근거로, 구조화된 퀴즈 딕셔너리를 생성한다.

    {"quiz_id", "type", "question", "choices", "answer",
     "explanation", "evidence", "source": {"file", "slide", "chunk_id"}}

핵심 설계 3가지:
  1) LLM에게 "JSON만 출력하라"고 강제하고, 코드에서 json.loads로 파싱한다.
  2) source(slide, chunk_id)는 LLM이 지어내게 두지 않고, retriever가 넘겨준
     실제 청크 메타데이터에서 코드가 직접 붙인다 (근거 신뢰성의 핵심).
  3) MODEL_NAME 한 줄(또는 load_generator 인자)만 바꾸면 모델 교체가 되도록 해서
     Qwen2.5-1.5B vs 3B vs 7B 비교 실험이 쉽도록 한다.
"""
import json
import re

# ---- 모델 교체는 여기 한 줄, 또는 load_generator(model_name=...) 인자로 ----
# RTX 4070(8GB) 기준 추천 실험 대상:
#   "Qwen/Qwen2.5-1.5B-Instruct"  # 베이스라인 (~3GB, 빠름)
#   "Qwen/Qwen2.5-3B-Instruct"    # 주력      (~6GB fp16)
#   "Qwen/Qwen2.5-7B-Instruct"    # 큰 모델   (8GB엔 quantize_4bit=True 필요)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


# LLM이 반드시 이 형식(JSON)만 뱉도록 강제하는 프롬프트.
# {n_choices}, {qtype_desc} 등은 호출 시 채워짐.
_JSON_RULES = """반드시 아래 JSON 형식 '하나'만 출력해라. 설명, 인사말, 코드블록 표시(```) 금지.
{{
  "type": "{qtype}",
  "question": "문제 (신입사원이 이해할 수 있게)",
  "choices": {choices_hint},
  "answer": 정답_보기의_인덱스(0부터 시작하는 정수),
  "explanation": "왜 그 답이 정답인지 해설",
  "evidence": "위 [참고자료]에서 정답의 근거가 된 문장을 그대로 인용",
  "source_ref": 근거가_나온_자료_번호(정수, 예: 1)
}}"""

SYSTEM_PROMPT_RAG = (
    "너는 신입사원 교육용 퀴즈 출제자다. 아래 [참고자료]에 있는 내용만 사용해서 "
    "퀴즈를 만들어라. 참고자료에 없는 내용은 절대 지어내지 마라. 만들 수 없으면 "
    'question에 "자료에 없음"이라고 써라.\n\n[참고자료]\n{context}\n\n'
    + _JSON_RULES
)

# baseline = RAG 없음 (참고자료를 주지 않음). RAG 효과를 대조하기 위한 실험군.
SYSTEM_PROMPT_BASELINE = (
    "너는 신입사원 교육용 퀴즈 출제자다. 아래 주제로 퀴즈를 만들어라.\n\n"
    + _JSON_RULES
)


def load_generator(model_name: str = MODEL_NAME, quantize_4bit: bool = False):
    """
    HuggingFace text-generation 파이프라인 로드.
    quantize_4bit=True 로 주면 7B 같은 큰 모델도 8GB VRAM에 올릴 수 있음
    (bitsandbytes 설치 필요: pip install bitsandbytes).
    """
    from transformers import pipeline

    model_kwargs = {}
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        import torch
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    return pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",       # GPU 자동 사용
        model_kwargs=model_kwargs,
    )


def _choices_hint(qtype: str) -> str:
    """퀴즈 유형별로 choices 형식 힌트를 준다."""
    if qtype == "true_false":       # OX 문제
        return '["O", "X"]'
    if qtype == "short_answer":     # 서술형/단답형
        return "[]"
    return '["보기1", "보기2", "보기3", "보기4"]'   # multiple_choice


def _build_context(retrieved: list[dict]) -> str:
    """검색된 청크들을 '자료 1', '자료 2' ... 로 번호 매겨 프롬프트용 문자열로 만든다.
    이 번호가 나중에 LLM의 source_ref와 매칭된다."""
    lines = []
    for i, r in enumerate(retrieved, start=1):
        lines.append(f"자료 {i}: {r['text']}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    """LLM 출력에서 첫 번째 JSON 객체를 뽑아 파싱한다.
    모델이 앞뒤에 잡소리나 ```json 펜스를 붙여도 { ... } 만 골라냄."""
    text = text.strip()
    # ```json ... ``` 펜스 제거
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        # 중괄호 균형을 맞춰 첫 완결 객체 추출
        start = text.find("{")
        if start == -1:
            raise ValueError("출력에 JSON 객체가 없음")
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            raise ValueError("JSON 중괄호가 닫히지 않음")
        candidate = text[start:end]
    return json.loads(candidate)


def _attach_source(quiz: dict, retrieved: list[dict], file_label: str) -> dict:
    """LLM이 고른 source_ref(자료 번호)를 실제 청크 메타데이터로 치환한다.
    -> source(slide, chunk_id)는 LLM이 아니라 검색 결과에서 나오므로 신뢰 가능."""
    ref = quiz.pop("source_ref", 1)
    try:
        idx = int(ref) - 1
    except (TypeError, ValueError):
        idx = 0
    if not (0 <= idx < len(retrieved)):
        idx = 0
    chunk = retrieved[idx]
    quiz["source"] = {
        "file": file_label,
        "slide": chunk["slide_no"],
        "chunk_id": chunk["chunk_id"],
    }
    return quiz


def _run(gen, messages, max_new_tokens: int, temperature: float) -> str:
    """공통 생성 호출. temperature=0 이면 그리디(재현성 O, 모델 비교에 적합)."""
    kwargs = {"max_new_tokens": max_new_tokens, "return_full_text": False}
    if temperature and temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        kwargs.update(do_sample=False)
    out = gen(messages, **kwargs)
    return out[0]["generated_text"] if isinstance(out[0]["generated_text"], str) \
        else out[0]["generated_text"][-1]["content"]


def generate_quiz(
    gen,
    topic: str,
    retrieved: list[dict] | None = None,
    qtype: str = "multiple_choice",
    quiz_id: str = "quiz-001",
    file_label: str = "자료 1",
    max_new_tokens: int = 400,
    temperature: float = 0.0,
) -> dict:
    """
    구조화된 퀴즈 딕셔너리 1개를 생성한다.
    - retrieved 가 주어지면 RAG(근거 기반), None 이면 baseline(모델 지식만).
    - 반환 실패 시(파싱 에러 등) _raw 필드에 원문을 담아 반환 -> 디버깅/성공률 집계용.
    """
    choices_hint = _choices_hint(qtype)

    if retrieved:
        system = SYSTEM_PROMPT_RAG.format(
            context=_build_context(retrieved), qtype=qtype, choices_hint=choices_hint
        )
    else:
        system = SYSTEM_PROMPT_BASELINE.format(qtype=qtype, choices_hint=choices_hint)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"주제: {topic}"},
    ]

    raw = _run(gen, messages, max_new_tokens, temperature)

    try:
        quiz = _extract_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        # 파싱 실패도 결과로 남긴다 (모델별 'JSON 성공률' 지표 계산에 사용)
        return {"quiz_id": quiz_id, "parse_ok": False, "error": str(e), "_raw": raw}

    quiz["quiz_id"] = quiz_id
    quiz["parse_ok"] = True
    if retrieved:
        quiz = _attach_source(quiz, retrieved, file_label)
    # dict 순서를 예시 형태에 맞춰 정리
    order = ["quiz_id", "type", "question", "choices", "answer",
             "explanation", "evidence", "source", "parse_ok"]
    return {k: quiz[k] for k in order if k in quiz}


if __name__ == "__main__":
    print("이 모듈은 HuggingFace 모델 다운로드가 필요합니다 (인터넷 환경).")
    print(f"현재 모델: {MODEL_NAME}")
    print("대화형 실행은  python generate_quiz.py  를 사용하세요.")

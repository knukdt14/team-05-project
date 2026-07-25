"""
src/generator_lc.py  [C 담당: LangChain 버전 퀴즈 생성기]

generator.py(순수 transformers)와 '똑같은 결과 dict'를 만들되, LangChain으로 구현.
장점 2가지:
  1) 모델 교체가 provider 인자 한 개로 끝남
       - provider="upstage" -> 업스테이지 Solar (API, 한국어 특화)
       - provider="hf"       -> 로컬 HuggingFace 모델 (Qwen2.5-1.5B/3B/7B ...)
     프롬프트/파싱/출처부착 코드는 그대로 재사용 -> 공정한 A/B 비교에 유리.
  2) Pydantic 스키마(Quiz)로 구조화 출력 -> "줄글 -> JSON dict" 변환이 안정적.

A(청킹)·B(검색) 코드는 전혀 건드리지 않는다. 여기서 받는 입력은 B의 retriever가
넘겨주는 list[dict] ({chunk_id, slide_no, text}) 뿐이다.
"""
from typing import List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

# generator.py(raw 버전)의 유틸을 그대로 재사용 -> 두 버전이 같은 파싱/출처 로직을 공유
from generator import _extract_json, _attach_source, _build_context


# ---------------------- 출력 스키마 (구조화 출력의 핵심) ----------------------
class Quiz(BaseModel):
    """LLM이 채워야 할 퀴즈 한 문항의 구조. 이 스키마가 프롬프트에 형식 지침으로 주입된다."""
    type: str = Field(description="multiple_choice / true_false / short_answer 중 하나")
    question: str = Field(description="신입사원이 이해할 수 있는 문제")
    choices: List[str] = Field(description="보기 목록. OX면 [\"O\",\"X\"], 서술형이면 []")
    answer: int = Field(description="정답 보기의 인덱스(0부터 시작). 서술형이면 0")
    explanation: str = Field(description="왜 그 답이 정답인지 해설")
    evidence: str = Field(description="[참고자료]에서 근거가 된 문장을 그대로 인용")
    source_ref: int = Field(default=1, description="근거가 나온 자료 번호(정수, 예: 1)")


# ---------------------- 모델 팩토리 (여기가 '모델 교체' 지점) ----------------------
def make_llm(provider: str = "upstage",
             model_name: str = "solar-pro",
             temperature: float = 0.0,
             max_new_tokens: int = 512):
    """
    provider에 따라 LangChain ChatModel을 반환한다.

    provider="upstage":
        환경변수 UPSTAGE_API_KEY 필요.  export UPSTAGE_API_KEY=...
        model_name 예: "solar-pro" (모델명/요금은 업스테이지 콘솔에서 확인 권장)
    provider="hf" (로컬):
        model_name 예: "Qwen/Qwen2.5-1.5B-Instruct" / "...-3B-Instruct"
        7B를 8GB VRAM에 올리려면 load_in_4bit 옵션 참고(아래 주석).
    """
    if provider == "upstage":
        from langchain_upstage import ChatUpstage
        return ChatUpstage(model=model_name, temperature=temperature)

    if provider == "hf":
        from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
        pipeline_kwargs = {"max_new_tokens": max_new_tokens, "return_full_text": False}
        if temperature > 0:
            pipeline_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            pipeline_kwargs["do_sample"] = False   # 그리디(재현성 O, 모델 비교에 적합)
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            device_map="auto",
            pipeline_kwargs=pipeline_kwargs,
            # 7B 양자화가 필요하면 아래처럼:
            # model_kwargs={"load_in_4bit": True},
        )
        return ChatHuggingFace(llm=llm)  # chat 템플릿(system/user 역할)을 적용해줌

    raise ValueError(f"알 수 없는 provider: {provider} (upstage / hf 중 선택)")


SYSTEM_RAG = (
    "너는 신입사원 교육용 퀴즈 출제자다. 아래 [참고자료]에 있는 내용만 사용해서 "
    "퀴즈를 만들어라. 참고자료에 없는 내용은 절대 지어내지 마라.\n\n"
    "[참고자료]\n{context}\n\n"
    "출력 형식은 아래 지침을 반드시 따라라(JSON 하나만):\n{format_instructions}"
)

SYSTEM_BASELINE = (
    "너는 신입사원 교육용 퀴즈 출제자다. 아래 주제로 퀴즈를 만들어라.\n\n"
    "출력 형식은 아래 지침을 반드시 따라라(JSON 하나만):\n{format_instructions}"
)


def build_generator(provider: str = "upstage",
                    model_name: str = "solar-pro",
                    temperature: float = 0.0,
                    use_rag: bool = True):
    """생성기(프롬프트+모델)를 준비해서 반환. 실험 시 이 인자만 바꾸면 됨."""
    llm = make_llm(provider, model_name, temperature)
    parser = PydanticOutputParser(pydantic_object=Quiz)
    system = SYSTEM_RAG if use_rag else SYSTEM_BASELINE
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "주제: {topic}\n퀴즈 유형: {qtype}"),
    ]).partial(format_instructions=parser.get_format_instructions())
    # chain: 프롬프트 -> LLM -> 문자열.  (JSON 파싱은 아래에서 직접 -> 성공률 집계 위해)
    chain = prompt | llm | StrOutputParser()
    return chain


def generate_quiz(chain,
                  topic: str,
                  retrieved: list[dict] | None = None,
                  qtype: str = "multiple_choice",
                  quiz_id: str = "quiz-001",
                  file_label: str = "자료 1") -> dict:
    """
    구조화된 퀴즈 dict 1개 생성.
    - retrieved 있으면 RAG, None이면 baseline(모델 지식만).
    - 파싱 실패 시 parse_ok=False로 반환 -> 모델별 'JSON 성공률' 지표 계산에 사용.
    """
    context = _build_context(retrieved) if retrieved else "(참고자료 없음)"
    raw = chain.invoke({"context": context, "topic": topic, "qtype": qtype})

    try:
        data = _extract_json(raw)
        quiz = Quiz.model_validate(data).model_dump()
    except Exception as e:
        return {"quiz_id": quiz_id, "parse_ok": False, "error": str(e), "_raw": raw}

    quiz["quiz_id"] = quiz_id
    quiz["parse_ok"] = True
    if retrieved:
        quiz = _attach_source(quiz, retrieved, file_label)  # source는 검색결과에서 부착
    else:
        quiz.pop("source_ref", None)

    order = ["quiz_id", "type", "question", "choices", "answer",
             "explanation", "evidence", "source", "parse_ok"]
    return {k: quiz[k] for k in order if k in quiz}


if __name__ == "__main__":
    print("대화형 실행은  python generate_quiz.py  를 사용하세요.")
    print("모델 교체 예:")
    print("  build_generator(provider='upstage', model_name='solar-pro')")
    print("  build_generator(provider='hf', model_name='Qwen/Qwen2.5-3B-Instruct')")

"""
src/generator.py

[샌드박스에서 실행 불가 - huggingface 모델 다운로드 필요]
Qwen2.5-1.5B-Instruct으로 baseline(RAG 없이) vs RAG(검색된 청크 포함) 답변을 생성.
실제 실행 환경(다른 AI/로컬 PC/GPU서버)에서 아래 코드를 그대로 쓰면 됨.
"""
from transformers import pipeline

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT_RAG = (
    "너는 신입사원 안전교육 퀴즈 출제자다. 아래 [참고자료]에 있는 내용만 사용해서 "
    "퀴즈 문제, 정답, 해설, 근거문장(evidence)을 만들어라. 참고자료에 없는 내용은 "
    "절대 지어내지 말고, 자료에 없으면 '자료에 없음'이라고 답해라.\n[참고자료]\n{context}"
)

SYSTEM_PROMPT_BASELINE = (
    "너는 신입사원 안전교육 퀴즈 출제자다. 아래 주제로 퀴즈 문제, 정답, 해설을 만들어라."
)


def load_generator():
    return pipeline(
        "text-generation",
        model=MODEL_NAME,
        device_map="auto",  # GPU 있으면 자동 사용, 없으면 CPU
    )


def generate_baseline(gen, topic: str, max_new_tokens: int = 300) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_BASELINE},
        {"role": "user", "content": f"주제: {topic}"},
    ]
    out = gen(messages, max_new_tokens=max_new_tokens)
    return out[0]["generated_text"][-1]["content"]


def generate_rag(gen, topic: str, retrieved_chunks: list[str], max_new_tokens: int = 300) -> str:
    context = "\n---\n".join(retrieved_chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_RAG.format(context=context)},
        {"role": "user", "content": f"주제: {topic}"},
    ]
    out = gen(messages, max_new_tokens=max_new_tokens)
    return out[0]["generated_text"][-1]["content"]


if __name__ == "__main__":
    print("이 모듈은 huggingface 모델 다운로드가 필요해 이 샌드박스에서는 실행되지 않음.")
    print(f"실행 환경에서: pip install transformers torch --break-system-packages")
    print(f"모델: {MODEL_NAME}")

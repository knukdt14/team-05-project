"""
src/verify_answers.py  [C 담당: 정답 정확성 검증 - 논문 dual-verification 이식]

생성된 퀴즈의 answer가 '진짜 맞는 답'인지 독립적으로 검증한다.
방법(AutoCode 논문 §5의 dual-verification을 퀴즈에 적용):
  ① 퀴즈의 answer를 가린다
  ② 검증자 LLM에게 [자료]를 근거로 그 문제를 '풀게' 한다 (정답 번호만)
  ③ 검증자가 고른 번호 == 원래 answer 이면 '정답 검증됨'

검증자 2명 (사용자 선택: 둘 다):
  - Qwen-3B (자체검증): 생성 모델과 동일 -> '자기 답을 자기가 맞다' 할 편향 가능
                        (논문 Finding 4: LLM 자기평가는 사람과 상관 ~0)
  - Solar   (교차검증): 다른 모델이 판정 -> 더 객관적

지표: answer_verified = (검증자 답 == 원래 answer) 비율
실행:  (.venv) cd src && python verify_answers.py
결과:  콘솔 표 + results/answer_verification.csv
"""
import re
import json
import gc
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from documents import extract_slide_texts
from chunking import chunk_documents
from generator_lc import make_llm
from experiment import EVAL_SET, PPTX_PATH

RESULTS_DIR = Path("../results")

# 검증할 퀴즈 파일 (2단계에서 만든 것)
QUIZ_FILES = {
    "qwen-3b+RAG":       RESULTS_DIR / "quizzes_qwen-3b_rag.json",
    "qwen-3b(baseline)": RESULTS_DIR / "quizzes_qwen-3b_base.json",
}

# 검증자 2명
VERIFIERS = [
    {"name": "Qwen자체검증", "provider": "hf",      "model": "Qwen/Qwen2.5-3B-Instruct"},
    {"name": "Solar교차검증", "provider": "upstage", "model": "solar-mini"},
]

VERIFY_SYSTEM = (
    "너는 시험 채점자다. 아래 [자료]에 있는 내용만 근거로 객관식 문제의 정답을 고른다. "
    "정답 보기의 번호(0부터 시작하는 정수) '하나'만 출력하라. 설명·다른 말 금지.\n\n"
    "[자료]\n{context}"
)


def build_verifier(provider: str, model: str):
    """검증자 체인: [자료]+문제 -> 정답 번호(텍스트)."""
    llm = make_llm(provider, model, temperature=0.0, max_new_tokens=8)
    prompt = ChatPromptTemplate.from_messages([
        ("system", VERIFY_SYSTEM),
        ("human", "문제: {question}\n보기:\n{choices}\n정답 번호:"),
    ])
    return prompt | llm | StrOutputParser()


def solve(chain, context: str, question: str, choices: list[str]):
    """검증자가 고른 정답 인덱스를 반환 (파싱 실패 시 None)."""
    choices_str = "\n".join(f"{i}) {c}" for i, c in enumerate(choices))
    raw = chain.invoke({"context": context, "question": question, "choices": choices_str})
    m = re.search(r"\d+", raw)
    return int(m.group()) if m else None


def main():
    print("슬라이드 원문 로딩 중 (검증자에게 줄 '진짜 자료')...")
    docs = extract_slide_texts(PPTX_PATH)
    chunks = chunk_documents(docs)
    slide_texts: dict[int, list[str]] = {}
    for c in chunks:
        slide_texts.setdefault(c["slide_no"], []).append(c["text"])

    rows = []
    for vf in VERIFIERS:
        print(f"\n▶ {vf['name']} ({vf['model']}) 로딩...")
        chain = build_verifier(vf["provider"], vf["model"])

        for label, path in QUIZ_FILES.items():
            if not path.exists():
                print(f"   [건너뜀: {path.name} 없음]")
                continue
            quizzes = json.loads(path.read_text(encoding="utf-8"))
            ok, total = 0, 0

            for quiz in quizzes:
                # 객관식(choices 있음)만 검증 가능
                if not quiz.get("parse_ok") or not quiz.get("choices"):
                    continue
                # quiz_id -> EVAL_SET 인덱스 (정답 슬라이드 = 검증용 진짜 자료)
                try:
                    idx = int(quiz["quiz_id"].split("-")[1]) - 1
                    case = EVAL_SET[idx]
                except (KeyError, ValueError, IndexError):
                    continue
                context = " ".join(t for s in case["relevant_slides"]
                                   for t in slide_texts.get(s, []))

                pred = solve(chain, context, quiz["question"], quiz["choices"])
                total += 1
                if pred is not None and pred == quiz.get("answer"):
                    ok += 1

            rate = round(ok / total, 3) if total else 0.0
            rows.append({"verifier": vf["name"], "quiz": label,
                         "verified": rate, "ok": ok, "n": total})
            print(f"   {label:20s} answer_verified={rate}  ({ok}/{total})")

        # 다음 검증자 로딩 전 VRAM 정리 (Qwen 로컬 모델용)
        del chain
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- 결과표 ----
    print("\n" + "=" * 58)
    print(f"{'퀴즈':20s} {'Qwen자체검증':>15s} {'Solar교차검증':>15s}")
    print("-" * 58)
    for label in QUIZ_FILES:
        q = next((r["verified"] for r in rows
                  if r["quiz"] == label and r["verifier"] == "Qwen자체검증"), "-")
        s = next((r["verified"] for r in rows
                  if r["quiz"] == label and r["verifier"] == "Solar교차검증"), "-")
        print(f"{label:20s} {str(q):>15s} {str(s):>15s}")
    print("=" * 58)

    # ---- CSV 저장 ----
    out = RESULTS_DIR / "answer_verification.csv"
    lines = ["verifier,quiz,answer_verified,ok,n\n"]
    for r in rows:
        lines.append(f"{r['verifier']},{r['quiz']},{r['verified']},{r['ok']},{r['n']}\n")
    out.write_text("".join(lines), encoding="utf-8")
    print(f"\n저장됨: {out}")


if __name__ == "__main__":
    main()

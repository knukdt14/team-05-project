"""
src/generate_quiz.py  [C 담당 진입점 - 대화형 퀴즈 생성]

실행하면 '주제'를 입력받아, 그 주제로 검색 -> 구조화된 퀴즈 dict를 출력한다.

흐름:
  ① (처음 한 번만) A+B: PPT를 청킹/임베딩해서 Chroma 인덱스 구축
  ② (주제 칠 때마다) C: 검색 -> LangChain LLM으로 퀴즈 생성 -> dict 출력

모델 교체는 아래 build_generator(...) 인자만 바꾸면 됨:
  provider="upstage", model_name="solar-pro"          # 업스테이지 Solar (UPSTAGE_API_KEY 필요)
  provider="hf",      model_name="Qwen/Qwen2.5-3B-Instruct"   # 로컬

주의: 업스테이지를 쓰려면 실행 전에 환경변수 설정
  (PowerShell)  $env:UPSTAGE_API_KEY="발급받은_키"
  (bash)        export UPSTAGE_API_KEY=발급받은_키
"""
import json

# .env 파일에서 UPSTAGE_API_KEY, HF_TOKEN 등을 자동으로 읽어옴 (프로젝트 루트의 .env)
from dotenv import load_dotenv
load_dotenv()

from documents import extract_slide_texts
from chunking import chunk_documents
from embeddings import ProductionEmbedder      # 실제 임베딩(e5-small). 빠른 오프라인 테스트는 LocalTestEmbedder
from index import build_index
from retriever import retrieve
from generator_lc import build_generator, generate_quiz

PPTX_PATH = "../data/deck.pptx"     # 프로젝트 data 폴더에 넣어둔 PPT
TOP_K = 3

# ---- 최종 확정 모델: 로컬 Qwen-3B (모델 비교 실험 + 사내정보 보안으로 선정) ----
PROVIDER = "hf"                              # 로컬 실행 (데이터 외부 유출 X)
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def main():
    print("① 인덱스 구축 중 (A+B: 청킹 + 임베딩 + Chroma)...")
    docs = extract_slide_texts(PPTX_PATH)
    chunks = chunk_documents(docs)
    embedder = ProductionEmbedder("intfloat/multilingual-e5-small")
    collection = build_index(chunks, embedder)
    print(f"   완료: 슬라이드 {len(docs)}개 -> 청크 {len(chunks)}개\n")

    print(f"② 생성기 준비: provider={PROVIDER}, model={MODEL_NAME}")
    chain = build_generator(provider=PROVIDER, model_name=MODEL_NAME, temperature=0.0)

    print("\n주제를 입력하면 퀴즈를 생성합니다. (그냥 엔터 치면 종료)\n")
    n = 1
    while True:
        topic = input(f"[{n}] 퀴즈 주제: ").strip()
        if not topic:
            print("종료합니다.")
            break

        retrieved = retrieve(collection, embedder, topic, k=TOP_K)
        quiz = generate_quiz(chain, topic, retrieved, quiz_id=f"quiz-{n:03d}")

        print(json.dumps(quiz, ensure_ascii=False, indent=2))
        if not quiz.get("parse_ok"):
            print("  ⚠️ JSON 파싱 실패 (모델 출력이 형식을 안 지킴). _raw 확인 필요.")
        print()
        n += 1


if __name__ == "__main__":
    main()

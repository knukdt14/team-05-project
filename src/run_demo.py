"""
run_demo.py
전체 파이프라인을 연결해서 실행하는 진입점.

이 샌드박스(HuggingFace 접속 차단)에서 실제로 동작 확인된 부분:
  - documents.py  (pptx 텍스트 추출)           OK
  - chunking.py   (chunk_size=150 청킹)         OK
  - index.py      (Chroma 인덱싱)               OK (LocalTestEmbedder로 대체)
  - retriever.py  (top-k 검색 + recall@k/MRR)   OK
  - evaluator.py  (grounded_score)              OK

인터넷 되는 환경에서 embedder만 교체하면 그대로 production 파이프라인이 됨:
  from embeddings import ProductionEmbedder
  embedder = ProductionEmbedder("intfloat/multilingual-e5-small")
  (LocalTestEmbedder -> ProductionEmbedder 로 한 줄만 바꾸면 됨)

generator.py(Qwen 생성)와 bert_score_eval은 모델 다운로드가 필요해 이 샌드박스에서는
실행하지 않고, 코드만 준비해둠.
"""
from documents import extract_slide_texts
from chunking import chunk_documents
from embeddings import LocalTestEmbedder
from index import build_index
from retriever import retrieve, recall_at_k, mrr, context_precision
from evaluator import grounded_score

PPTX_PATH = "../../deck.pptx"

# 안전관리 섹션 기준 질문-정답 세트 (실제 슬라이드 번호가 정답)
EVAL_SET = [
    {"query": "크레인 작업 시 위험발생요인과 대책은?", "relevant_slides": {38}},
    {"query": "프레스 작업 중 재해사례는 언제 어디서 발생했나?", "relevant_slides": {39}},
    {"query": "하인리히 법칙에서 중대재해와 경미한 재해의 비율은?", "relevant_slides": {36}},
    {"query": "지게차 작업 시 위험요소와 대책은?", "relevant_slides": {37}},
    {"query": "물질안전보건자료(MSDS)에 명시되는 내용은?", "relevant_slides": {40}},
]


def main():
    print("=" * 60)
    print("1) 문서 추출 + 청킹 (chunk_size=150)")
    print("=" * 60)
    docs = extract_slide_texts(PPTX_PATH)
    chunks = chunk_documents(docs)
    print(f"슬라이드 {len(docs)}개 -> 청크 {len(chunks)}개\n")

    print("=" * 60)
    print("2) 임베딩 + Chroma 인덱싱")
    print("=" * 60)
    print("[주의] 아래는 LocalTestEmbedder(TF-IDF)를 사용한 결과입니다.")
    print("       실제 의미기반 검색 품질이 아니라 '배관 검증'용입니다.")
    embedder = LocalTestEmbedder()
    embedder.fit([c["text"] for c in chunks])
    collection = build_index(chunks, embedder)
    print(f"인덱싱 완료: {collection.count()}개 청크\n")

    print("=" * 60)
    print("3) 검색 평가 (recall@3, MRR, context_precision@3)")
    print("=" * 60)
    recalls, mrrs, precisions = [], [], []
    for case in EVAL_SET:
        retrieved = retrieve(collection, embedder, case["query"], k=3)
        slides = [r["slide_no"] for r in retrieved]
        r = recall_at_k(slides, case["relevant_slides"])
        m = mrr(slides, case["relevant_slides"])
        p = context_precision(slides, case["relevant_slides"])
        recalls.append(r); mrrs.append(m); precisions.append(p)
        print(f"- {case['query'][:35]:35s} | recall@3={r:.2f} MRR={m:.2f} "
              f"precision@3={p:.2f} | 검색결과 슬라이드 {slides}")

    print(f"\n평균 recall@3={sum(recalls)/len(recalls):.3f}  "
          f"평균 MRR={sum(mrrs)/len(mrrs):.3f}  "
          f"평균 precision@3={sum(precisions)/len(precisions):.3f}")

    print("\n" + "=" * 60)
    print("4) grounded_score 예시 (모델 없이 순수 텍스트 연산)")
    print("=" * 60)
    sample_chunks = [c["text"] for c in chunks if c["slide_no"] == 39]
    good_evidence = "2015년 7월 부산 기장군 자동차부품공장에서 프레스 작업 중 협착 사고 발생"
    bad_evidence = "2022년 인천에서 화재사고 발생"
    print("실제근거 기반 답변 grounded_score:", round(grounded_score(good_evidence, sample_chunks), 3))
    print("지어낸 답변 grounded_score      :", round(grounded_score(bad_evidence, sample_chunks), 3))

    print("\n" + "=" * 60)
    print("5) 여기부터는 이 샌드박스에서 실행 불가 (huggingface 접속 차단)")
    print("=" * 60)
    print("- embeddings.ProductionEmbedder (intfloat/multilingual-e5-small)")
    print("- generator.py (Qwen2.5-1.5B-Instruct 로 baseline/RAG 답변 생성)")
    print("- evaluator.bert_score_eval (BERTScore)")
    print("-> 인터넷 되는 환경에서 위 3개 모듈만 그대로 이어붙이면 전체 파이프라인 완성")


if __name__ == "__main__":
    main()

"""
verify_chunk_size_sensitivity.py

실제 프로덕션에서 쓰는 청킹 전략(main.py의 CHUNKING_STRATEGY)으로,
현재 pipeline_config.json에 확정된 top_k/fetch_k가 여전히 최적인지
진짜 bge-m3 모델로 재검증하는 스크립트.

배경: B의 파라미터 최적화(top_k=5, fetch_k=15)는 chunk_size=150(recursive)
청킹 기준으로 실험된 것인데, main.py가 실제로는 chunk_size=300
(sentence_pack) 청킹을 쓰고 있어서, 청크 크기가 바뀌면 최적 파라미터도
바뀔 수 있다는 우려가 있었음. 이 스크립트는 그 우려를 실제 모델로 확인함.

실행 방법:
    conda activate <프로젝트 환경>
    python verify_chunk_size_sensitivity.py
"""
from __future__ import annotations

import sys
import time

from chunking import chunk_documents
from documents import extract_slide_structures, extract_slide_texts
from embeddings import BGEEmbedder
from retriever import retrieve_diverse, recall_at_k, reciprocal_rank, diversity_at_k
from settings import PROJECT_ROOT, load_config
from vector_store import build_index

# main.py와 반드시 동일하게 맞출 것 (여기가 실제 프로덕션에서 쓰는 전략)
CHUNKING_STRATEGY = "sentence_pack"

PPTX_PATH = PROJECT_ROOT / "data" / r"C:\team-05-project-merge-integrated\경북대 교육 발표자료 250416-1.pptx"
EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval"

sys.path.append(str(EVAL_SET_PATH))
from ground_truth_quiz import ground_truth_quizzes  # noqa: E402


def load_documents(strategy: str):
    return (
        extract_slide_structures(PPTX_PATH)
        if strategy == "title_body"
        else extract_slide_texts(PPTX_PATH)
    )


def evaluate(collection, embedder, k: int, fetch_k: int) -> dict:
    recalls, mrrs, diversities = [], [], []
    t0 = time.time()
    for q in ground_truth_quizzes:
        retrieved = retrieve_diverse(collection, embedder, q["question"], k=k, fetch_k=fetch_k)
        slides = [item["slide_no"] for item in retrieved]
        relevant = {q["source"]["slide"]}
        recalls.append(recall_at_k(slides, relevant))
        mrrs.append(reciprocal_rank(slides, relevant))
        diversities.append(diversity_at_k(slides, k))
    elapsed = time.time() - t0
    n = len(ground_truth_quizzes)
    return {
        "recall": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        "diversity": sum(diversities) / n,
        "sec_per_query": elapsed / n,
    }


def print_row(label: str, r: dict) -> None:
    print(
        "{:18s} | recall={:.3f}  MRR={:.3f}  diversity={:.3f}  {:.3f}초/문항".format(
            label, r["recall"], r["mrr"], r["diversity"], r["sec_per_query"]
        )
    )


def main() -> None:
    config = load_config()
    current_top_k = int(config["top_k"])
    current_fetch_k = int(config["fetch_k"])

    print("=" * 80)
    print(f"청킹 전략: {CHUNKING_STRATEGY} (main.py와 반드시 동일해야 함)")
    print(f"현재 pipeline_config.json 설정: top_k={current_top_k}, fetch_k={current_fetch_k}")
    print("=" * 80)

    documents = load_documents(CHUNKING_STRATEGY)
    chunks = chunk_documents(documents, CHUNKING_STRATEGY, document_id=config["document_id"])
    lengths = [len(c["text"]) for c in chunks]
    print(f"슬라이드 {len(documents)}개 -> 청크 {len(chunks)}개 "
          f"(평균 {sum(lengths)/len(lengths):.1f}자, 최대 {max(lengths)}자)")

    print(f"\n임베딩 모델 로딩: {config['embedding_model']}")
    embedder = BGEEmbedder(config["embedding_model"])

    print("\n인덱싱 중...")
    collection = build_index(
        chunks, embedder,
        collection_name=f"verify_{CHUNKING_STRATEGY}",
        persist_dir=PROJECT_ROOT / "results" / "chroma" / f"verify_{CHUNKING_STRATEGY}",
        space=config["distance"],
    )
    print(f"인덱싱 완료: {collection.count()}개 청크\n")

    print("=" * 80)
    print("실험 A: top-k 스윕 (fetch_k=15 고정)")
    print("=" * 80)
    for k in [3, 5, 10, 15]:
        r = evaluate(collection, embedder, k=k, fetch_k=15)
        marker = "  <- 현재 설정" if k == current_top_k else ""
        print_row(f"k={k}" + marker, r)

    print("\n" + "=" * 80)
    print("실험 B: fetch_k 스윕 (k=5 고정)")
    print("=" * 80)
    for fetch_k in [7, 10, 15, 20, 30]:
        r = evaluate(collection, embedder, k=5, fetch_k=fetch_k)
        marker = "  <- 현재 설정" if fetch_k == current_fetch_k else ""
        print_row(f"fetch_k={fetch_k}" + marker, r)

    print("\n" + "=" * 80)
    print("결론")
    print("=" * 80)
    current = evaluate(collection, embedder, k=current_top_k, fetch_k=current_fetch_k)
    best_k = max([3, 5, 10, 15], key=lambda k: evaluate(collection, embedder, k=k, fetch_k=15)["mrr"])
    print(f"현재 설정(k={current_top_k}, fetch_k={current_fetch_k}) 실측: "
          f"recall={current['recall']:.3f}, MRR={current['mrr']:.3f}, "
          f"diversity={current['diversity']:.3f}")
    print(f"MRR 기준 최적 top-k: {best_k}")
    if best_k == current_top_k:
        print("-> 현재 pipeline_config.json 설정이 sentence_pack 청킹에서도 최적입니다. 변경 불필요.")
    else:
        print(f"-> 현재 설정과 다름! top_k를 {best_k}로 바꾸는 것을 검토하세요.")


if __name__ == "__main__":
    main()

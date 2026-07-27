"""
grid_search.py

청킹 전략(A) x 임베딩 모델(B) 전체 조합(4 x 5 = 20개)을
공용 정답셋(49문항)으로 평가해서 최적 조합을 찾는 격자 실험.

실행 시간을 아끼기 위해, 임베딩 모델을 바깥 루프로 두고 "한 번만 로딩"한 뒤,
그 안에서 청킹 전략 4개를 바꿔가며 인덱싱+평가만 반복함
(모델 로딩 5번, 인덱싱 20번).

실행 방법:
    conda activate <프로젝트 환경>
    python grid_search.py
"""
from __future__ import annotations

import sys
import time

from chunking import chunk_documents, SUPPORTED_STRATEGIES
from documents import extract_slide_structures, extract_slide_texts
from retriever import retrieve_diverse, recall_at_k, reciprocal_rank, diversity_at_k
from settings import PROJECT_ROOT, load_config
from vector_store import build_index

PPTX_PATH = PROJECT_ROOT / "data" / r"C:\team-05-project-merge-integrated\경북대 교육 발표자료 250416-1.pptx"
EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval"
sys.path.append(str(EVAL_SET_PATH))
from ground_truth_quiz import ground_truth_quizzes  # noqa: E402

K = 5
FETCH_K = 15

# 임베딩 모델 후보 (paraphrase-MiniLM, ko-sroberta는 이미 성능 낮음이 확인돼서 제외,
# 필요하면 아래 리스트에 다시 추가하면 됨)
MODEL_CANDIDATES = [
    {"name": "BAAI/bge-m3", "use_prefix": False},
    {"name": "dragonkue/multilingual-e5-small-ko-v2", "use_prefix": True},
    {"name": "intfloat/multilingual-e5-small", "use_prefix": True},
]


class FlexibleEmbedder:
    def __init__(self, model_name: str, use_prefix: bool):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.use_prefix = use_prefix
        print(f"  로딩 완료 (device={device})")

    def embed_documents(self, texts):
        inputs = [f"passage: {t}" for t in texts] if self.use_prefix else texts
        return self.model.encode(inputs, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        q = f"query: {text}" if self.use_prefix else text
        return self.model.encode(q, normalize_embeddings=True).tolist()


def load_documents_for(strategy: str):
    return (
        extract_slide_structures(PPTX_PATH)
        if strategy == "title_body"
        else extract_slide_texts(PPTX_PATH)
    )


def evaluate(collection, embedder) -> dict:
    recalls, mrrs, diversities = [], [], []
    for q in ground_truth_quizzes:
        retrieved = retrieve_diverse(collection, embedder, q["question"], k=K, fetch_k=FETCH_K)
        slides = [item["slide_no"] for item in retrieved]
        relevant = {q["source"]["slide"]}
        recalls.append(recall_at_k(slides, relevant))
        mrrs.append(reciprocal_rank(slides, relevant))
        diversities.append(diversity_at_k(slides, K))
    n = len(ground_truth_quizzes)
    return {
        "recall": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        "diversity": sum(diversities) / n,
    }


def main() -> None:
    config = load_config()
    results = []

    # 청킹 전략별 문서/청크는 모델과 무관하므로 미리 한 번씩만 준비
    print("=" * 90)
    print("청킹 전략별 문서 추출 + 청킹 준비")
    print("=" * 90)
    chunks_by_strategy = {}
    for strategy in SUPPORTED_STRATEGIES:
        docs = load_documents_for(strategy)
        chunks = chunk_documents(docs, strategy, document_id=config["document_id"])
        chunks_by_strategy[strategy] = chunks
        lengths = [len(c["text"]) for c in chunks]
        print(f"  {strategy:15s}: 청크 {len(chunks)}개 (평균 {sum(lengths)/len(lengths):.1f}자)")

    print("\n" + "=" * 90)
    print(f"격자 실험 시작: 임베딩모델 {len(MODEL_CANDIDATES)}개 x 청킹전략 {len(SUPPORTED_STRATEGIES)}개 "
          f"= {len(MODEL_CANDIDATES) * len(SUPPORTED_STRATEGIES)}개 조합")
    print("=" * 90)

    for cand in MODEL_CANDIDATES:
        model_name = cand["name"]
        print(f"\n임베딩 모델 로딩: {model_name}")
        t0 = time.time()
        embedder = FlexibleEmbedder(model_name, cand["use_prefix"])
        print(f"  (로딩 시간: {time.time()-t0:.1f}초)")

        for strategy in SUPPORTED_STRATEGIES:
            chunks = chunks_by_strategy[strategy]
            t0 = time.time()
            collection = build_index(
                chunks, embedder,
                collection_name=f"grid_{strategy}_{model_name.replace('/', '_')}",
                persist_dir=PROJECT_ROOT / "results" / "chroma" / "grid_search",
                space=config["distance"],
            )
            index_time = time.time() - t0

            r = evaluate(collection, embedder)
            r.update({"model": model_name, "strategy": strategy, "index_sec": index_time})
            results.append(r)
            print(f"  [{strategy:15s}] recall={r['recall']:.3f}  MRR={r['mrr']:.3f}  "
                  f"diversity={r['diversity']:.3f}  (인덱싱 {index_time:.1f}초)")

    print("\n" + "=" * 90)
    print("전체 결과 (MRR 내림차순 정렬)")
    print("=" * 90)
    results.sort(key=lambda r: r["mrr"], reverse=True)
    print(f"{'순위':>4} | {'모델':45s} | {'청킹전략':15s} | {'recall':>7} | {'MRR':>6} | {'diversity':>9}")
    print("-" * 100)
    for rank, r in enumerate(results, start=1):
        marker = "  <- 1위" if rank == 1 else ""
        print(f"{rank:>4} | {r['model']:45s} | {r['strategy']:15s} | "
              f"{r['recall']:>7.3f} | {r['mrr']:>6.3f} | {r['diversity']:>9.3f}{marker}")

    best = results[0]
    print(f"\n최종 추천 조합: 임베딩={best['model']}, 청킹={best['strategy']}")
    print(f"현재 pipeline_config.json/main.py 설정과 비교해서 다르면 변경을 검토하세요.")


if __name__ == "__main__":
    main()

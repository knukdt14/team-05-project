"""
src/optimize_bge_m3.py
선정된 BAAI/bge-m3 모델 하나만 놓고 파라미터를 스윕하며 최적화.

실험 A: top-k 스윕 (fetch_k=15 고정) - k=3/5/10/15
실험 B: fetch_k 스윕 (k=5 고정) - fetch_k=10/15/20/30
실험 C: 거리계산 방식 비교 (k=5, fetch_k=15 고정) - l2 vs cosine

실행 방법:
    conda activate DL_PY311
    cd src
    python optimize_bge_m3.py
"""
import sys
import time
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "..", "data", "eval"))

from documents import extract_slide_texts
from chunking import chunk_documents
from index import build_index
from retriever import retrieve_diverse, recall_at_k, mrr, context_precision
from ground_truth_quiz import ground_truth_quizzes

PPTX_PATH = os.path.join(THIS_DIR, "..", "..", r"C:\team-05-project\src\경북대 교육 발표자료 250416-1.pptx")
MODEL_NAME = "BAAI/bge-m3"  # bge-m3는 query:/passage: 접두사 불필요


class Embedder:
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()


def diversity_at_k(slides, k):
    return len(set(slides)) / k


def evaluate(collection, embedder, k, fetch_k):
    recalls, mrrs, precisions, diversities = [], [], [], []
    t0 = time.time()
    for q in ground_truth_quizzes:
        retrieved = retrieve_diverse(collection, embedder, q["question"], k=k, fetch_k=fetch_k)
        slides = [r["slide_no"] for r in retrieved]
        relevant = {q["source"]["slide"]}
        recalls.append(recall_at_k(slides, relevant))
        mrrs.append(mrr(slides, relevant))
        precisions.append(context_precision(slides, relevant))
        diversities.append(diversity_at_k(slides, k))
    elapsed = time.time() - t0
    n = len(ground_truth_quizzes)
    return {
        "recall": sum(recalls) / n, "mrr": sum(mrrs) / n,
        "precision": sum(precisions) / n, "diversity": sum(diversities) / n,
        "sec_per_query": elapsed / n,
    }


def print_row(label, r):
    print("{:25s} | recall={:.3f}  MRR={:.3f}  precision={:.3f}  diversity={:.3f}  "
          "{:.3f}초/문항".format(label, r["recall"], r["mrr"], r["precision"],
                              r["diversity"], r["sec_per_query"]))


def main():
    print("=" * 90)
    print("bge-m3 파라미터 최적화 (문항 {}개)".format(len(ground_truth_quizzes)))
    print("=" * 90)
    docs = extract_slide_texts(PPTX_PATH)
    chunks = chunk_documents(docs)
    print("슬라이드 {}개 -> 청크 {}개".format(len(docs), len(chunks)))

    print("\n모델 로딩: {}".format(MODEL_NAME))
    embedder = Embedder(MODEL_NAME)

    # L2 기준 인덱스 (실험 A, B에서 재사용)
    t0 = time.time()
    collection_l2 = build_index(
        chunks, embedder,
        persist_dir=os.path.join(THIS_DIR, "chroma_opt_l2"),
        collection_name="opt_l2", space="l2",
    )
    print("인덱싱(L2) 완료: {:.1f}초\n".format(time.time() - t0))

    # ---------------- 실험 A: top-k 스윕 ----------------
    print("=" * 90)
    print("실험 A: top-k 스윕 (fetch_k=15 고정)")
    print("=" * 90)
    for k in [3, 5, 10, 15]:
        r = evaluate(collection_l2, embedder, k=k, fetch_k=15)
        print_row("k={}".format(k), r)

    # ---------------- 실험 B: fetch_k 스윕 ----------------
    print("\n" + "=" * 90)
    print("실험 B: fetch_k 스윕 (k=5 고정)")
    print("=" * 90)
    for fetch_k in [7, 10, 15, 20, 30]:
        r = evaluate(collection_l2, embedder, k=5, fetch_k=fetch_k)
        print_row("fetch_k={}".format(fetch_k), r)

    # ---------------- 실험 C: 거리계산 방식 비교 ----------------
    print("\n" + "=" * 90)
    print("실험 C: 거리계산 방식 비교 (k=5, fetch_k=15)")
    print("=" * 90)
    r_l2 = evaluate(collection_l2, embedder, k=5, fetch_k=15)
    print_row("space=l2", r_l2)

    t0 = time.time()
    collection_cos = build_index(
        chunks, embedder,
        persist_dir=os.path.join(THIS_DIR, "chroma_opt_cosine"),
        collection_name="opt_cosine", space="cosine",
    )
    print("인덱싱(cosine) 완료: {:.1f}초".format(time.time() - t0))
    r_cos = evaluate(collection_cos, embedder, k=5, fetch_k=15)
    print_row("space=cosine", r_cos)

    if r_l2["recall"] == r_cos["recall"] and r_l2["mrr"] == r_cos["mrr"]:
        print("\n-> l2와 cosine 결과가 동일함 (정규화된 벡터라 이론대로 순위가 같게 나옴)")
    else:
        print("\n-> l2와 cosine 결과가 다름! (예상과 달라 원인 확인 필요)")


if __name__ == "__main__":
    main()

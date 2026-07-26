"""
src/compare_models_before.py
[수정 전 버전] retrieve() 기준으로 5개 임베딩 모델을 비교. top-k=5.

실행 방법:
    conda activate DL_PY311
    cd src
    python compare_models_before.py
"""
import sys
import time
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "..", "data", "eval"))

from documents import extract_slide_texts
from chunking import chunk_documents
from index import build_index
from retriever import retrieve, recall_at_k, mrr, context_precision
from ground_truth_quiz import ground_truth_quizzes

PPTX_PATH = os.path.join(THIS_DIR, "..", "..", r"C:\team-05-project\src\경북대 교육 발표자료 250416-1.pptx")

K = 5  # top-k 값

CANDIDATES = [
    {"name": "dragonkue/multilingual-e5-small-ko-v2", "use_prefix": True},
    {"name": "intfloat/multilingual-e5-small", "use_prefix": True},
    {"name": "BAAI/bge-m3", "use_prefix": False},
    {"name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "use_prefix": False},
    {"name": "jhgan/ko-sroberta-multitask", "use_prefix": False},
]


class FlexibleEmbedder:
    def __init__(self, model_name, use_prefix=True):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.use_prefix = use_prefix

    def embed_documents(self, texts):
        inputs = [f"passage: {t}" for t in texts] if self.use_prefix else texts
        return self.model.encode(inputs, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        q = f"query: {text}" if self.use_prefix else text
        return self.model.encode(q, normalize_embeddings=True).tolist()


def diversity_at_k(slides, k):
    return len(set(slides)) / k


def evaluate(collection, embedder, k):
    recalls, mrrs, precisions, diversities = [], [], [], []
    t0 = time.time()
    for q in ground_truth_quizzes:
        retrieved = retrieve(collection, embedder, q["question"], k=k)
        slides = [r["slide_no"] for r in retrieved]
        relevant = {q["source"]["slide"]}
        recalls.append(recall_at_k(slides, relevant))
        mrrs.append(mrr(slides, relevant))
        precisions.append(context_precision(slides, relevant))
        diversities.append(diversity_at_k(slides, k))
    elapsed = time.time() - t0
    n = len(ground_truth_quizzes)
    return {
        "recall": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        "precision": sum(precisions) / n,
        "diversity": sum(diversities) / n,
        "sec_per_query": elapsed / n,
    }


def print_table(title, results_dict, k):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    header = "{:50s} | {:>9} | {:>6} | {:>11} | {:>11} | {:>8}".format(
        "모델", "recall@{}".format(k), "MRR", "precision@{}".format(k),
        "diversity@{}".format(k), "초/질문")
    print(header)
    print("-" * 100)
    for name, r in results_dict.items():
        line = "{:50s} | {:>9.3f} | {:>6.3f} | {:>11.3f} | {:>11.3f} | {:>8.3f}".format(
            name, r["recall"], r["mrr"], r["precision"], r["diversity"], r["sec_per_query"])
        print(line)


def main():
    print("=" * 100)
    print("[수정 전] 문서 추출 + 청킹 (chunk_size=150) / 정답셋 {}문항 / k={}".format(
        len(ground_truth_quizzes), K))
    print("=" * 100)
    docs = extract_slide_texts(PPTX_PATH)
    chunks = chunk_documents(docs)
    print("슬라이드 {}개 -> 청크 {}개".format(len(docs), len(chunks)))
    print()

    results = {}
    for cand in CANDIDATES:
        model_name = cand["name"]
        print("=" * 100)
        print("모델: " + model_name)
        print("=" * 100)
        embedder = FlexibleEmbedder(model_name, use_prefix=cand["use_prefix"])
        collection = build_index(
            chunks, embedder,
            persist_dir=os.path.join(THIS_DIR, "chroma_before_" + model_name.replace("/", "_")),
            collection_name="ajin_eval_before",
        )
        results[model_name] = evaluate(collection, embedder, k=K)
        r = results[model_name]
        print("recall@{}={:.3f}  MRR={:.3f}  precision@{}={:.3f}  diversity@{}={:.3f}".format(
            K, r["recall"], r["mrr"], K, r["precision"], K, r["diversity"]))
        print()

    print_table("[[ 수정 전 (retrieve) - 5개 모델 비교, k={} ]]".format(K), results, K)


if __name__ == "__main__":
    main()

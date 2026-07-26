"""Build and compare the four requested PowerPoint chunking strategies.

This script keeps the embedder, Chroma retrieval, top-k, and evaluation queries
fixed so that only the chunking strategy changes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from statistics import mean

from chunking import CHUNKING_CONFIGS, build_all_chunk_sets
from documents import extract_slide_structures, extract_slide_texts


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = PROJECT_ROOT / "eval_questions_100.json"


def _validate_eval_set(eval_set: list[dict]) -> list[dict]:
    """Validate and normalize retrieval evaluation records."""
    normalized = []
    for index, case in enumerate(eval_set, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"평가 항목 #{index}은 JSON 객체여야 합니다.")
        query = str(case.get("query", "")).strip()
        relevant_slides = case.get("relevant_slides")
        if not query:
            raise ValueError(f"평가 항목 #{index}의 query가 비어 있습니다.")
        if not isinstance(relevant_slides, list) or not relevant_slides:
            raise ValueError(
                f"평가 항목 #{index}의 relevant_slides는 비어 있지 않은 배열이어야 합니다."
            )
        normalized.append(
            {
                **case,
                "eval_id": str(case.get("eval_id", f"eval-{index:03d}")),
                "query": query,
                "relevant_slides": [int(slide) for slide in relevant_slides],
                "category": str(case.get("category", "미분류")),
            }
        )
    return normalized


def _read_eval_file(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        if path.suffix.lower() == ".jsonl":
            data = [json.loads(line) for line in file if line.strip()]
        else:
            data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("평가 파일의 최상위 값은 JSON 배열이어야 합니다.")
    return _validate_eval_set(data)


# 다른 모듈에서도 ``from compare_chunking import DEFAULT_EVAL_SET``으로 사용할 수 있다.
DEFAULT_EVAL_SET = _read_eval_file(DEFAULT_EVAL_FILE)


def load_eval_set(path: str | None) -> list[dict]:
    if path is None:
        return DEFAULT_EVAL_SET
    return _read_eval_file(Path(path))


def chunk_statistics(chunks: list[dict]) -> dict[str, float | int]:
    lengths = [len(chunk["text"]) for chunk in chunks]
    return {
        "chunks": len(chunks),
        "avg_chars": mean(lengths) if lengths else 0.0,
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
    }


def evaluate_strategy(
    strategy: str,
    chunks: list[dict],
    eval_set: list[dict],
    persist_root: Path,
    top_k: int,
    embedding_model: str | None,
) -> dict:
    try:
        from embeddings import LocalTestEmbedder, ProductionEmbedder
        from index import build_index
        from retriever import context_precision, mrr, recall_at_k, retrieve
    except ModuleNotFoundError as exc:
        if exc.name == "chromadb":
            raise RuntimeError(
                "Chroma is not installed. Run 'pip install -r requirements.txt' "
                "or use --stats-only."
            ) from exc
        raise

    if embedding_model:
        embedder = ProductionEmbedder(embedding_model)
    else:
        embedder = LocalTestEmbedder()
        embedder.fit([chunk["text"] for chunk in chunks])

    collection_dir = persist_root / strategy
    if collection_dir.exists():
        shutil.rmtree(collection_dir)

    collection = build_index(
        chunks,
        embedder,
        persist_dir=str(collection_dir),
        collection_name=f"chunking_{strategy}",
    )

    case_results = []
    for case in eval_set:
        retrieved = retrieve(collection, embedder, case["query"], k=top_k)
        retrieved_slides = [item["slide_no"] for item in retrieved]
        relevant_slides = set(case["relevant_slides"])
        case_results.append(
            {
                "eval_id": case["eval_id"],
                "category": case["category"],
                "query": case["query"],
                "relevant_slides": sorted(relevant_slides),
                "retrieved_slides": retrieved_slides,
                "retrieved_chunk_ids": [item["chunk_id"] for item in retrieved],
                f"recall@{top_k}": recall_at_k(retrieved_slides, relevant_slides),
                "mrr": mrr(retrieved_slides, relevant_slides),
                f"context_precision@{top_k}": context_precision(
                    retrieved_slides, relevant_slides
                ),
            }
        )

    recall_key = f"recall@{top_k}"
    precision_key = f"context_precision@{top_k}"
    return {
        "strategy": strategy,
        "config": CHUNKING_CONFIGS[strategy],
        "evaluation_questions": len(eval_set),
        **chunk_statistics(chunks),
        recall_key: mean(result[recall_key] for result in case_results),
        "mrr": mean(result["mrr"] for result in case_results),
        precision_key: mean(result[precision_key] for result in case_results),
        "cases": case_results,
    }


def print_summary(results: list[dict], top_k: int) -> None:
    recall_key = f"recall@{top_k}"
    precision_key = f"context_precision@{top_k}"
    header = (
        f"{'strategy':14s} {'chunks':>7s} {'avg chars':>9s} "
        f"{recall_key:>10s} {'MRR':>7s} {precision_key:>20s}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result['strategy']:14s} {result['chunks']:7d} "
            f"{result['avg_chars']:9.1f} {result[recall_key]:10.3f} "
            f"{result['mrr']:7.3f} {result[precision_key]:20.3f}"
        )


def print_chunk_stats(results: list[dict]) -> None:
    header = (
        f"{'strategy':14s} {'chunks':>7s} {'avg chars':>9s} "
        f"{'min':>7s} {'max':>7s}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result['strategy']:14s} {result['chunks']:7d} "
            f"{result['avg_chars']:9.1f} {result['min_chars']:7d} "
            f"{result['max_chars']:7d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pptx", help="Path to the source PPTX file")
    parser.add_argument(
        "--eval-file",
        help=(
            "JSON or JSONL retrieval evaluation set "
            f"(default: {DEFAULT_EVAL_FILE.name})"
        ),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--embedding-model",
        help=(
            "SentenceTransformer model name. Omit to use the offline TF-IDF "
            "test embedder."
        ),
    )
    parser.add_argument("--persist-root", default=".chunking_compare")
    parser.add_argument("--output", help="Optional JSON result path")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Compare chunk counts and lengths without Chroma or an embedder.",
    )
    args = parser.parse_args()

    flat_slides = extract_slide_texts(args.pptx)
    structured_slides = extract_slide_structures(args.pptx)
    chunk_sets = build_all_chunk_sets(flat_slides, structured_slides)
    if args.stats_only:
        results = [
            {
                "strategy": strategy,
                "config": CHUNKING_CONFIGS[strategy],
                **chunk_statistics(chunks),
            }
            for strategy, chunks in chunk_sets.items()
        ]
        print_chunk_stats(results)
    else:
        eval_set = load_eval_set(args.eval_file)
        persist_root = Path(args.persist_root)
        results = [
            evaluate_strategy(
                strategy,
                chunks,
                eval_set,
                persist_root,
                args.top_k,
                args.embedding_model,
            )
            for strategy, chunks in chunk_sets.items()
        ]   
        print_summary(results, args.top_k)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()

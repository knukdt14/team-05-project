"""고정 질의로 모든 청킹 전략을 비교하고 검색 청크를 저장한다."""

from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path
from typing import Any

from chunking import CHUNKING_CONFIGS, SUPPORTED_STRATEGIES
from main import (
    PPTX_PATH,
    QUIZ_TYPE,
    TOPIC_COUNT,
    TOPIC_SEED,
    VALIDATE_CHOICES,
    prepare_pipeline,
)
from quiz_generator import generate_quiz, load_llm
from retriever import retrieve_diverse
from settings import PROJECT_ROOT, load_config
from topic import extract_topics_llm, sample_topics


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "chunking_comparison"
DEFAULT_QUERY_FILE = PROJECT_ROOT / "data" / "evaluation_queries.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "같은 고정 질의로 여러 청킹 전략의 퀴즈를 생성하고, "
            "각 퀴즈에 검색된 청크 원문을 저장합니다."
        )
    )
    parser.add_argument("--pptx", type=Path, default=PPTX_PATH)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=SUPPORTED_STRATEGIES,
        default=list(SUPPORTED_STRATEGIES),
        help="기본값: 지원하는 모든 청킹 전략",
    )
    parser.add_argument("--topic-count", type=int, default=TOPIC_COUNT)
    parser.add_argument("--topic-seed", type=int, default=TOPIC_SEED)
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERY_FILE if DEFAULT_QUERY_FILE.exists() else None,
        help="고정 질의 JSON. 기본값: data/evaluation_queries.json",
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=None,
        help="고정 질의 파일에서 사용할 질의 수",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument(
        "--validate-choices",
        action=argparse.BooleanOptionalAction,
        default=VALIDATE_CHOICES,
    )
    return parser.parse_args(argv)


def load_fixed_queries(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("고정 질의 JSON은 배열이어야 합니다.")
    queries = []
    for index, item in enumerate(data, start=1):
        if isinstance(item, str):
            query = item.strip()
            item = {"query_id": f"q-{index:03d}", "query": query}
        elif isinstance(item, dict):
            query = str(item.get("query", "")).strip()
            item = dict(item)
            item["query_id"] = str(item.get("query_id") or f"q-{index:03d}")
            item["query"] = query
        else:
            raise ValueError(f"{index}번째 질의는 문자열 또는 객체여야 합니다.")
        if not query:
            raise ValueError(f"{index}번째 질의가 비어 있습니다.")
        if "relevant_slides" in item:
            item["relevant_slides"] = [
                int(slide) for slide in item["relevant_slides"]
            ]
        queries.append(item)
    if not queries:
        raise ValueError("고정 질의가 없습니다.")
    return queries


def _retrieval_metrics(
    query: dict[str, Any],
    retrieved: list[dict[str, Any]],
) -> dict[str, float] | None:
    relevant = query.get("relevant_slides")
    if not relevant:
        return None
    relevant = {int(slide) for slide in relevant}
    retrieved_slides = [int(chunk["slide_no"]) for chunk in retrieved]
    hits = [slide for slide in retrieved_slides if slide in relevant]
    first_hit = next(
        (rank for rank, slide in enumerate(retrieved_slides, start=1)
         if slide in relevant),
        None,
    )
    return {
        "recall_at_k": len(set(hits)) / len(relevant),
        "precision_at_k": len(set(hits)) / len(retrieved_slides)
        if retrieved_slides else 0.0,
        "mrr": 1.0 / first_hit if first_hit else 0.0,
    }


def _mean_metrics(metrics: list[dict[str, Any]]) -> dict[str, float] | None:
    if not metrics:
        return None
    keys = [
        key
        for key, value in metrics[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {
        key: sum(item[key] for item in metrics) / len(metrics)
        for key in keys
    }


def _retrieved_chunks_for_output(
    retrieved: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "slide_no": int(chunk["slide_no"]),
            "title": str(chunk.get("title", "")),
            "distance": float(chunk["distance"]),
            "text": str(chunk["text"]),
        }
        for rank, chunk in enumerate(retrieved, start=1)
    ]


def _print_retrieved_chunks(
    strategy: str,
    topic: str,
    retrieved_chunks: list[dict[str, Any]],
) -> None:
    print(f"\n[{strategy}] 고정 질의: {topic}")
    for chunk in retrieved_chunks:
        print(
            f"\n  검색 {chunk['rank']}위 | slide {chunk['slide_no']} | "
            f"{chunk['chunk_id']} | distance={chunk['distance']:.4f}"
        )
        print("  " + chunk["text"].replace("\n", "\n  "))


def _save_json(value: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _empty_accelerator_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _new_experiment(
    strategy: str,
    queries: list[dict[str, Any]],
    *,
    chunk_size: int,
    overlap: int,
) -> dict[str, Any]:
    return {
        "chunking": {
            "strategy": strategy,
            "chunk_size": chunk_size,
            "overlap": overlap,
        },
        "queries": queries,
        "topics": [item["query"] for item in queries],
        "quizzes": [],
        "failures": [],
        "retrieval_metrics": [],
        "retrieval_metrics_mean": None,
    }


def run_strategy(
    *,
    strategy: str,
    queries: list[dict[str, Any]],
    pptx_path: Path,
    output_dir: Path,
    generator: Any,
    max_attempts: int,
    validate_choices: bool,
) -> tuple[dict[str, Any], Path]:
    chunk_config = CHUNKING_CONFIGS[strategy]
    chunk_size = int(chunk_config["chunk_size"])
    overlap = int(chunk_config["overlap"])
    experiment_id = f"{strategy}_{chunk_size}_{overlap}"
    output_path = output_dir / f"{experiment_id}.json"
    experiment = _new_experiment(
        strategy,
        queries,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    config, embedder, collection = prepare_pipeline(
        pptx_path,
        strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    try:
        for index, query_item in enumerate(queries, start=1):
            query_id = str(query_item["query_id"])
            query = str(query_item["query"])
            retrieved_output: list[dict[str, Any]] = []
            try:
                retrieved = retrieve_diverse(
                    collection,
                    embedder,
                    query,
                    k=int(config["top_k"]),
                    fetch_k=int(config["fetch_k"]),
                )
                retrieved_output = _retrieved_chunks_for_output(retrieved)
                _print_retrieved_chunks(strategy, query, retrieved_output)
                retrieval_metrics = _retrieval_metrics(query_item, retrieved)
                if retrieval_metrics is not None:
                    experiment["retrieval_metrics"].append(
                        {"query_id": query_id, "query": query, **retrieval_metrics}
                    )

                quiz = generate_quiz(
                    generator,
                    query,
                    retrieved,
                    quiz_id=f"{strategy}-{query_id}",
                    quiz_type=QUIZ_TYPE,
                    max_attempts=max_attempts,
                    file_label=config["file_label"],
                    validate_choices=validate_choices,
                )
                quiz["topic"] = query
                quiz["query_id"] = query_id
                quiz["query"] = query
                quiz["chunking"] = dict(experiment["chunking"])
                quiz["retrieved_chunks"] = retrieved_output
                experiment["quizzes"].append(quiz)
                print(f"\n  생성 완료: {quiz['quiz_id']} - {quiz['question']}")
            except Exception as error:  # noqa: BLE001 - 다음 주제 실험을 계속 진행
                failure = {
                    "quiz_id": f"{strategy}-{query_id}",
                    "query_id": query_id,
                    "query": query,
                    "error": str(error),
                    "retrieved_chunks": retrieved_output,
                    "traceback": traceback.format_exc(),
                }
                experiment["failures"].append(failure)
                print(f"\n  생성 실패: {failure['quiz_id']} - {error!r}")
            finally:
                print(f"  중간 저장: {_save_json(experiment, output_path)}")
        experiment["retrieval_metrics_mean"] = _mean_metrics(
            experiment["retrieval_metrics"]
        )
        _save_json(experiment, output_path)
    finally:
        del collection
        del embedder
        _empty_accelerator_cache()

    return experiment, output_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.pptx.exists():
        raise FileNotFoundError(f"PPTX 파일이 없습니다: {args.pptx}")
    if args.topic_count <= 0:
        raise ValueError("topic-count는 1 이상이어야 합니다.")
    if args.query_count is not None and args.query_count <= 0:
        raise ValueError("query-count는 1 이상이어야 합니다.")
    if args.max_attempts <= 0:
        raise ValueError("max-attempts는 1 이상이어야 합니다.")

    config = load_config()
    print(f"생성 모델 로딩: {config['llm_model']}")
    generator = load_llm(config["llm_model"])

    if args.queries is not None:
        if not args.queries.exists():
            raise FileNotFoundError(f"고정 질의 파일이 없습니다: {args.queries}")
        queries = load_fixed_queries(args.queries)
        if args.query_count is not None:
            queries = queries[:args.query_count]
        if not queries:
            raise ValueError("사용할 고정 질의가 없습니다.")
    else:
        topic_pool = extract_topics_llm(args.pptx, generator, use_cache=True)
        topics = sample_topics(
            topic_pool,
            n=args.topic_count,
            seed=args.topic_seed,
        )
        if not topics:
            raise ValueError("추출된 주제가 없습니다. topic.py를 확인하세요.")
        queries = [
            {"query_id": f"q-{index:03d}", "query": topic}
            for index, topic in enumerate(topics, start=1)
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"비교 고정 질의 {len(queries)}개: {[q['query'] for q in queries]}")
    print(f"청킹 전략: {args.strategies}")

    experiments = []
    for strategy in args.strategies:
        print(f"\n{'=' * 72}")
        print(f"청킹 실험 시작: {strategy}")
        print("=" * 72)
        experiment, output_path = run_strategy(
            strategy=strategy,
            queries=queries,
            pptx_path=args.pptx,
            output_dir=args.output_dir,
            generator=generator,
            max_attempts=args.max_attempts,
            validate_choices=args.validate_choices,
        )
        experiments.append(experiment)
        print(
            f"{strategy} 완료: 퀴즈 {len(experiment['quizzes'])}개, "
            f"실패 {len(experiment['failures'])}개"
        )
        print(f"결과: {output_path}")

    combined = {
        "pptx": str(args.pptx),
        "topic_seed": args.topic_seed,
        "queries": queries,
        "topics": [item["query"] for item in queries],
        "experiments": experiments,
    }
    combined_path = _save_json(
        combined,
        args.output_dir / "all_chunking_quizzes.json",
    )
    print(f"\n전체 비교 결과: {combined_path}")


if __name__ == "__main__":
    main()

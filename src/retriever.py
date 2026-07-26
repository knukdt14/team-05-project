"""
src/retriever.py
top-k 검색 + retrieval 단계 평가지표(recall@k, MRR, context precision)
"""


def retrieve(collection, embedder, query: str, k: int = 3):
    q_emb = embedder.embed_query(query)
    result = collection.query(query_embeddings=[q_emb], n_results=k)
    return [
        {"chunk_id": cid, "text": doc, "slide_no": meta["slide_no"]}
        for cid, doc, meta in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0]
        )
    ]


def retrieve_diverse(collection, embedder, query: str, k: int = 3, fetch_k: int = 15):
    """
    이슈 수정판: 같은 슬라이드의 청크가 top-k 안에 중복으로 들어가는 문제를 해결.
    - 일단 넉넉하게 fetch_k개(기본 15개)를 가져온 다음
    - 슬라이드 번호가 같으면 그중 가장 점수 높은(=순서상 먼저 나온) 청크만 남기고
    - 서로 다른 슬라이드 k개가 채워질 때까지만 채택
    이러면 top-k 안의 "실질적으로 서로 다른 정보"의 개수가 늘어나서
    precision@k가 개선됨 (recall@k는 유지하면서).
    """
    q_emb = embedder.embed_query(query)
    result = collection.query(query_embeddings=[q_emb], n_results=fetch_k)

    seen_slides = set()
    diverse = []
    for cid, doc, meta in zip(result["ids"][0], result["documents"][0], result["metadatas"][0]):
        slide_no = meta["slide_no"]
        if slide_no in seen_slides:
            continue
        seen_slides.add(slide_no)
        diverse.append({"chunk_id": cid, "text": doc, "slide_no": slide_no})
        if len(diverse) >= k:
            break
    return diverse


# ---------------------- Retrieval 평가지표 ----------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """정답 청크 중 top-k 안에 몇 %가 들어왔는지."""
    if not relevant_ids:
        return 0.0
    hit = len(set(retrieved_ids) & relevant_ids)
    return hit / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank: 정답이 몇 번째 순위에 처음 등장하는지의 역수.
    recall@k는 '들어왔는지 여부'만 보는데, MRR은 '얼마나 상위에 나왔는지'까지 반영함
    -> recall@k와 짝을 이루는 순위 품질 지표로 추천."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def context_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """검색된 top-k 중 실제로 관련있는 것의 비율 (recall의 반대 짝 지표).
    recall@k = '정답을 놓치지 않았는가', context_precision = '가져온 것 중 쓸모없는
    게 얼마나 섞였는가' -> 청크사이즈/overlap 실험에서 이 둘을 같이 보면
    '더 크게 자를수록 recall은 오르지만 precision은 떨어진다' 같은 트레이드오프를
    보여줄 수 있어서 추천."""
    if not retrieved_ids:
        return 0.0
    hit = len(set(retrieved_ids) & relevant_ids)
    return hit / len(retrieved_ids)


if __name__ == "__main__":
    from documents import extract_slide_texts
    from chunking import chunk_documents
    from embeddings import LocalTestEmbedder
    from index import build_index

    docs = extract_slide_texts("../../deck.pptx")
    chunks = chunk_documents(docs)
    embedder = LocalTestEmbedder()
    embedder.fit([c["text"] for c in chunks])
    collection = build_index(chunks, embedder)

    # 안전관리 섹션(슬라이드 38~39)에서 실제 질문-정답 예시
    test_cases = [
        {
            "query": "크레인 작업 시 위험발생요인과 대책은?",
            "relevant_slides": {38},
        },
        {
            "query": "프레스 작업 중 재해사례는 언제 어디서 발생했나?",
            "relevant_slides": {39},
        },
        {
            "query": "하인리히 법칙에서 중대재해와 경미한 재해의 비율은?",
            "relevant_slides": {36},
        },
    ]

    print(f"{'질문':40s} | recall@3 | MRR   | precision@3")
    print("-" * 80)
    for case in test_cases:
        retrieved = retrieve(collection, embedder, case["query"], k=3)
        retrieved_slides = [r["slide_no"] for r in retrieved]
        relevant = case["relevant_slides"]

        r = recall_at_k(retrieved_slides, relevant)
        m = mrr(retrieved_slides, relevant)
        p = context_precision(retrieved_slides, relevant)
        print(f"{case['query'][:38]:40s} | {r:.2f}     | {m:.2f}  | {p:.2f}")
        print(f"   -> 검색된 슬라이드: {retrieved_slides}")

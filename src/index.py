"""
src/index.py
청크들을 임베딩해서 Chroma 컬렉션에 저장한다.
embedder는 embeddings.py의 ProductionEmbedder 또는 LocalTestEmbedder를 주입받는다
(의존성 주입 방식 -> 나중에 실제 모델로 교체할 때 이 파일은 안 건드려도 됨).
"""
import chromadb


def build_index(chunks: list[dict], embedder, persist_dir: str = "./chroma_store",
                 collection_name: str = "ajin_safety_quiz", space: str = "l2"):
    """
    space: Chroma가 유사도를 계산하는 거리방식.
        - "l2"(기본값): 유클리드 거리
        - "cosine": 코사인 유사도 기반 거리
        - "ip": 내적(inner product) 기반
    우리는 embed_documents/embed_query에서 이미 normalize_embeddings=True로
    벡터를 정규화해뒀기 때문에, 이론적으로는 l2와 cosine의 "순위"가 동일해야 함
    -> 실제로 동일한지 실험으로 검증하는 용도로 이 파라미터를 노출함.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name, metadata={"hnsw:space": space})

    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_documents(texts)

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"slide_no": c["slide_no"]} for c in chunks],
    )
    return collection


if __name__ == "__main__":
    from documents import extract_slide_texts
    from chunking import chunk_documents
    from embeddings import LocalTestEmbedder

    docs = extract_slide_texts("../../deck.pptx")
    chunks = chunk_documents(docs)

    embedder = LocalTestEmbedder()
    embedder.fit([c["text"] for c in chunks])

    collection = build_index(chunks, embedder)
    print(f"Chroma 컬렉션 생성 완료: {collection.count()}개 청크 저장됨")

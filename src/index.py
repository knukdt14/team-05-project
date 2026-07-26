"""
src/index.py
청크들을 임베딩해서 Chroma 컬렉션에 저장한다.
embedder는 embeddings.py의 ProductionEmbedder 또는 LocalTestEmbedder를 주입받는다
(의존성 주입 방식 -> 나중에 실제 모델로 교체할 때 이 파일은 안 건드려도 됨).
"""
import chromadb


def build_index(
    chunks: list[dict],
    embedder,
    persist_dir: str = "./chroma_store",
    collection_name: str = "ajin_safety_quiz",
    space: str = "l2",
):
    """청크를 임베딩해 지정한 거리 방식의 Chroma 컬렉션을 만든다."""
    client = chromadb.PersistentClient(path=persist_dir)
    # 재실행 시 중복 방지
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        collection_name,
        metadata={"hnsw:space": space},
    )

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

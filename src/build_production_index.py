"""
src/build_production_index.py

B가 실험으로 확정한 최종 설정으로 "진짜 사용할" 벡터DB(Chroma 인덱스)를
한 번 구축해두는 스크립트. C(생성)나 A(통합)가 이 결과물을 그대로
가져다 쓰면 됨 -- 매번 새로 인덱싱할 필요 없이, 이 스크립트를 한 번만
실행해두면 됨.

최종 확정 설정 (B_최종보고서.md 참고):
    - 임베딩 모델: BAAI/bge-m3
    - 거리계산 방식: l2 (기본값, cosine과 결과 동일 확인됨)
    - top-k: 5, fetch_k: 15  (검색 시점에 retrieve_diverse()로 적용)

실행 방법:
    conda activate DL_PY311
    cd src
    python build_production_index.py

실행 후 생성되는 것:
    src/chroma_production/  <- 이 폴더가 "진짜" 벡터DB
    (이 폴더는 용량이 커서 .gitignore에 의해 깃에는 안 올라감.
     각자 로컬에서 이 스크립트를 한 번 실행해서 만들어 써야 함 --
     또는 팀 공유 드라이브에 압축해서 올려두고 다운받아 써도 됨)
"""
import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "..", "data", "eval"))

from documents import extract_slide_texts
from chunking import chunk_documents
from index import build_index

PPTX_PATH = os.path.join(THIS_DIR, "..", "..", r"C:\team-05-project\src\경북대 교육 발표자료 250416-1.pptx")
PRODUCTION_DIR = os.path.join(THIS_DIR, "chroma_production")
COLLECTION_NAME = "ajin_production"

MODEL_NAME = "BAAI/bge-m3"  # 최종 확정 모델


class BgeM3Embedder:
    """C/A가 검색할 때도 그대로 가져다 쓸 수 있는, bge-m3 전용 임베더."""

    def __init__(self, model_name: str = MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()


def main():
    print("=" * 80)
    print("최종 프로덕션 벡터DB 구축 시작")
    print("=" * 80)

    print("1) PPT 텍스트 추출 + 청킹")
    docs = extract_slide_texts(PPTX_PATH)
    chunks = chunk_documents(docs)
    print(f"   슬라이드 {len(docs)}개 -> 청크 {len(chunks)}개")

    print(f"\n2) 임베딩 모델 로딩: {MODEL_NAME}")
    embedder = BgeM3Embedder(MODEL_NAME)

    print(f"\n3) 인덱싱 (space=l2, 저장 위치: {PRODUCTION_DIR})")
    collection = build_index(
        chunks, embedder,
        persist_dir=PRODUCTION_DIR,
        collection_name=COLLECTION_NAME,
        space="l2",
    )
    print(f"   완료: {collection.count()}개 청크 저장됨")

    print("\n" + "=" * 80)
    print("완료! 검색할 때는 이렇게 불러오면 됨:")
    print("=" * 80)
    print("""
import chromadb
from build_production_index import BgeM3Embedder
from retriever import retrieve_diverse

client = chromadb.PersistentClient(path="chroma_production")
collection = client.get_collection("ajin_production")
embedder = BgeM3Embedder()

results = retrieve_diverse(collection, embedder, "질문 내용", k=5, fetch_k=15)
""")


if __name__ == "__main__":
    main()

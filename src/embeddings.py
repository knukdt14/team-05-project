"""
src/embeddings.py

[프로덕션 임베딩 모델 추천]
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 대신,
아래 모델을 추천함:

    intfloat/multilingual-e5-small

추천 이유 (속도 + 적합성):
- 파라미터 118M으로 multilingual-MiniLM-L12-v2(118M)와 비슷한 크기지만,
  MTEB(임베딩 벤치마크) 다국어 리트리벌 점수가 더 높음 (E5 계열은 검색 특화 학습)
- "e5" 계열은 애초에 검색(retrieval) 목적으로 대조학습(contrastive learning)되어
  RAG의 retriever 용도에 더 적합 (MiniLM-paraphrase는 문장유사도 범용 모델)
- small 버전은 CPU에서도 실시간에 가깝게 동작할 만큼 가벼움
- 사용 시 주의: e5 계열은 입력 앞에 "query: " 또는 "passage: " 접두사를 붙여야
  성능이 제대로 나옴 (아래 코드에 반영함)

[샌드박스 제약사항]
이 환경은 huggingface.co 접속이 막혀있어(x-deny-reason: host_not_allowed)
실제 모델 다운로드가 불가능함. 그래서 아래 두 가지를 구분해서 제공함:
  1) ProductionEmbedder: 실제 운영 환경(다른 AI/로컬 PC)에서 그대로 쓸 코드
  2) LocalTestEmbedder: 이 샌드박스에서 파이프라인 로직 자체를 검증하기 위한
     TF-IDF 기반 대체 임베딩 (모델 다운로드 불필요, sklearn만 사용)
     -> 실제 의미 검색 품질은 낮지만, "청킹->인덱싱->검색->평가"가 배관상
        올바르게 연결되는지 확인하는 용도로만 사용
"""


class ProductionEmbedder:
    """실제 운영 환경에서 사용할 임베딩 함수. (이 샌드박스에서는 실행 불가)"""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        return self.model.encode(prefixed, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(f"query: {text}", normalize_embeddings=True).tolist()


class LocalTestEmbedder:
    """
    샌드박스 로컬 검증용 TF-IDF 임베딩.
    실제 의미 기반 임베딩이 아니라 단어 빈도 기반이라 성능은 훨씬 낮지만,
    모델 다운로드 없이 파이프라인 배관(인덱싱->검색->평가) 검증에는 충분함.
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        # 한국어는 공백 기준 tokenizer로는 부족하지만, 로컬 검증 목적상 문자 n-gram 사용
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._fitted = False

    def fit(self, corpus: list[str]):
        self.vectorizer.fit(corpus)
        self._fitted = True

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            self.fit(texts)
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.vectorizer.transform([text]).toarray()[0].tolist()

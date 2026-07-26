"""검색 평가에 사용하는 임베딩 구현.

최종 선별 모델은 ``BAAI/bge-m3``이다. BGE-M3는 정규화 임베딩을 사용하고
E5 계열과 달리 ``query:``/``passage:`` 접두사를 붙이지 않는다.
``LocalTestEmbedder``는 모델 없이 배관만 점검하는 TF-IDF 대체 구현이며,
최종 청킹 선별에는 사용하지 않는다.
"""


class ProductionEmbedder:
    """실제 운영 환경에서 사용할 SentenceTransformer 임베딩 함수."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_prefix: bool | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.use_prefix = (
            "e5" in model_name.lower() if use_prefix is None else use_prefix
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        inputs = [f"passage: {text}" for text in texts] if self.use_prefix else texts
        return self.model.encode(inputs, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        query = f"query: {text}" if self.use_prefix else text
        return self.model.encode(query, normalize_embeddings=True).tolist()


class LocalTestEmbedder:
    """로컬 배관 검증용 TF-IDF 임베딩.

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

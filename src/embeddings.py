"""Final embedding model selected by the embedding/retrieval experiments."""

from __future__ import annotations


FINAL_EMBEDDING_MODEL = "BAAI/bge-m3"


class BGEEmbedder:
    """Normalized dense embeddings from BAAI/bge-m3.

    The embedding team found that BGE-M3 does not need the E5-style
    ``query:``/``passage:`` prefixes, so raw text is encoded here.
    """

    def __init__(
        self,
        model_name: str = FINAL_EMBEDDING_MODEL,
        *,
        batch_size: int = 16,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Windows에서 중단된 pytorch_model.bin 캐시로 인한 zip 손상을 피하고,
        # pickle을 사용하지 않는 safetensors 가중치를 우선 사용한다.
        self.model = SentenceTransformer(
            model_name,
            model_kwargs={"use_safetensors": True},
            device=self.device,
        )
        print(f"Embedding device: {self.device}")

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self.batch_size,
        ).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def to(self, device: str) -> None:
        """Move the model so the LLM can use the released accelerator memory."""
        self.model.to(device)
        self.device = device

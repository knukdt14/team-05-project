"""Upstage Solar Embedding API adapter with a persistent local cache."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import requests


UPSTAGE_EMBEDDING_ENDPOINT = "https://api.upstage.ai/v1/solar/embeddings"
PASSAGE_MODEL = "solar-embedding-1-large-passage"
QUERY_MODEL = "solar-embedding-1-large-query"


class UpstageEmbedder:
    """Embed passages and queries with the matching Upstage API models.

    Successful responses are cached by model name and text hash. Rebuilding a
    Chroma collection therefore does not charge for the same text twice.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_path: str | Path | None = None,
        query_cache_path: str | Path | None = None,
        request_interval: float = 0.3,
        timeout: float = 30.0,
        max_retries: int = 5,
        cache_write_retries: int = 8,
    ) -> None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        self.api_key = api_key or os.environ.get("UPSTAGE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "UPSTAGE_API_KEY가 없습니다. 프로젝트 루트의 .env 또는 "
                "환경변수에 키를 설정하세요."
            )

        self.cache_path = Path(cache_path) if cache_path else None
        if query_cache_path is not None:
            self.query_cache_path = Path(query_cache_path)
        elif self.cache_path is not None:
            self.query_cache_path = self.cache_path.with_name(
                f"{self.cache_path.stem}_queries{self.cache_path.suffix}"
            )
        else:
            self.query_cache_path = None
        self.request_interval = max(0.0, request_interval)
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_write_retries = max(1, cache_write_retries)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._passage_cache = self._load_cache(self.cache_path)
        self._query_cache = self._load_cache(self.query_cache_path)

    @staticmethod
    def _load_cache(path: Path | None) -> dict[str, list[float]]:
        if path is None or not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _cache_key(text: str, model: str) -> str:
        value = f"{model}\0{text}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    def _cache_for_model(
        self,
        model: str,
    ) -> tuple[dict[str, list[float]], Path | None]:
        if model == QUERY_MODEL:
            return self._query_cache, self.query_cache_path
        return self._passage_cache, self.cache_path

    def _save_cache(
        self,
        cache: dict[str, list[float]],
        path: Path | None,
    ) -> bool:
        """Atomically save a cache without making cache I/O pipeline-critical."""
        if path is None:
            return True

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(cache, ensure_ascii=False),
                encoding="utf-8",
            )
            for attempt in range(self.cache_write_retries):
                try:
                    os.replace(temporary, path)
                    return True
                except PermissionError as error:
                    if attempt + 1 >= self.cache_write_retries:
                        print(
                            f"    [캐시 경고] Windows가 {path.name} 교체를 "
                            f"{self.cache_write_retries}회 거부했습니다: {error}. "
                            "이번 임베딩은 메모리에서 계속 사용합니다."
                        )
                        return False
                    delay = min(2.0, 0.2 * (attempt + 1))
                    print(
                        f"    [캐시 재시도] {path.name} 잠금, "
                        f"{delay:.1f}초 대기 "
                        f"({attempt + 1}/{self.cache_write_retries})"
                    )
                    time.sleep(delay)
                except OSError as error:
                    print(
                        f"    [캐시 경고] {path} 저장 실패: {error}. "
                        "이번 임베딩은 메모리에서 계속 사용합니다."
                    )
                    return False
        except OSError as error:
            print(
                f"    [캐시 경고] 임시 캐시 파일 저장 실패: {error}. "
                "이번 임베딩은 메모리에서 계속 사용합니다."
            )
            return False
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return False

    @staticmethod
    def _retry_delay(response: requests.Response | None, attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    pass
        return float(2**attempt)

    def _embed(self, text: str, model: str) -> list[float]:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("빈 문자열은 임베딩할 수 없습니다.")

        cache, cache_path = self._cache_for_model(model)
        cache_key = self._cache_key(clean_text, model)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            response: requests.Response | None = None
            try:
                response = requests.post(
                    UPSTAGE_EMBEDDING_ENDPOINT,
                    headers=self.headers,
                    json={"input": clean_text, "model": model},
                    timeout=self.timeout,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    delay = self._retry_delay(response, attempt)
                    print(
                        f"    [Upstage 재시도] HTTP {response.status_code}, "
                        f"{delay:.1f}초 대기 ({attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                payload: dict[str, Any] = response.json()
                embedding = payload["data"][0]["embedding"]
                if not isinstance(embedding, list) or not embedding:
                    raise ValueError("Upstage 응답에 유효한 embedding이 없습니다.")
                vector = [float(value) for value in embedding]
                cache[cache_key] = vector
                self._save_cache(cache, cache_path)
                return vector
            except (requests.RequestException, KeyError, TypeError, ValueError) as error:
                last_error = error
                if attempt + 1 >= self.max_retries:
                    break
                delay = self._retry_delay(response, attempt)
                print(
                    f"    [Upstage 재시도] {error!r}, "
                    f"{delay:.1f}초 대기 ({attempt + 1}/{self.max_retries})"
                )
                time.sleep(delay)

        raise RuntimeError(
            f"Upstage 임베딩 API가 {self.max_retries}회 실패했습니다."
        ) from last_error

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        total = len(texts)
        for index, text in enumerate(texts, start=1):
            vectors.append(self._embed(text, PASSAGE_MODEL))
            if index % 50 == 0 or index == total:
                print(f"  Upstage 문서 임베딩: {index}/{total}")
            if self.request_interval and index < total:
                time.sleep(self.request_interval)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, QUERY_MODEL)

    def to(self, device: str) -> None:
        """Cloud embeddings have no local device to move."""

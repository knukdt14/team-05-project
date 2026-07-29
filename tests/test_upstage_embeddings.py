from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from upstage_embeddings import (  # noqa: E402
    PASSAGE_MODEL,
    QUERY_MODEL,
    UpstageEmbedder,
)


class UpstageEmbedderTests(unittest.TestCase):
    def test_uses_separate_passage_and_query_models_and_cache(self) -> None:
        response = Mock()
        response.status_code = 200
        response.headers = {}
        response.raise_for_status.return_value = None
        response.json.side_effect = [
            {"data": [{"embedding": [1, 2]}]},
            {"data": [{"embedding": [3, 4]}]},
        ]

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "embeddings.json"
            with patch("upstage_embeddings.requests.post", return_value=response) as post:
                embedder = UpstageEmbedder(
                    "test-key",
                    cache_path=cache_path,
                    request_interval=0,
                )
                self.assertEqual(embedder.embed_documents(["문서"]), [[1.0, 2.0]])
                self.assertEqual(embedder.embed_query("질문"), [3.0, 4.0])
                self.assertEqual(embedder.embed_documents(["문서"]), [[1.0, 2.0]])

            self.assertEqual(post.call_count, 2)
            self.assertEqual(
                post.call_args_list[0].kwargs["json"]["model"],
                PASSAGE_MODEL,
            )
            self.assertEqual(
                post.call_args_list[1].kwargs["json"]["model"],
                QUERY_MODEL,
            )
            self.assertTrue(cache_path.exists())
            self.assertTrue(
                cache_path.with_name("embeddings_queries.json").exists()
            )

    def test_retries_windows_cache_lock_without_failing_embedding(self) -> None:
        response = Mock()
        response.status_code = 200
        response.headers = {}
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": [{"embedding": [1, 2]}]}

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "embeddings.json"
            with (
                patch("upstage_embeddings.requests.post", return_value=response),
                patch(
                    "upstage_embeddings.os.replace",
                    side_effect=[PermissionError("locked"), None],
                ) as replace,
                patch("upstage_embeddings.time.sleep") as sleep,
            ):
                embedder = UpstageEmbedder(
                    "test-key",
                    cache_path=cache_path,
                    request_interval=0,
                )
                vector = embedder.embed_query("질문")

            self.assertEqual(vector, [1.0, 2.0])
            self.assertEqual(replace.call_count, 2)
            self.assertEqual(sleep.call_args_list, [call(0.2)])

    def test_continues_when_cache_remains_locked(self) -> None:
        response = Mock()
        response.status_code = 200
        response.headers = {}
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": [{"embedding": [1, 2]}]}

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "embeddings.json"
            with (
                patch("upstage_embeddings.requests.post", return_value=response),
                patch(
                    "upstage_embeddings.os.replace",
                    side_effect=PermissionError("locked"),
                ),
                patch("upstage_embeddings.time.sleep"),
            ):
                embedder = UpstageEmbedder(
                    "test-key",
                    cache_path=cache_path,
                    request_interval=0,
                    cache_write_retries=2,
                )
                vector = embedder.embed_query("질문")

            self.assertEqual(vector, [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()

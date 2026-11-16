import itertools
import zlib
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient

from shop_agent.ingest.loader import build_chunks, load_products, load_reviews
from shop_agent.retrieval.store import VectorStore

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


class FakeEmbedder:
    """모델을 내려받지 않는 테스트용 임베더.

    문자 bigram을 해시해서 벡터를 만든다. 글자가 많이 겹치는 문장끼리 가까워지므로
    검색 흐름이 제대로 연결됐는지 확인하기에는 충분하다.
    """

    dim = 256

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for a, b in itertools.pairwise(t):
                # hash()는 실행마다 값이 바뀌므로, 항상 같은 값을 주는 crc32를 쓴다
                out[i, zlib.crc32((a + b).encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


@pytest.fixture
def chunks():
    products = load_products(SAMPLE_DIR / "products.jsonl")
    reviews = load_reviews(SAMPLE_DIR / "reviews.jsonl")
    return build_chunks(products, reviews)


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def store():
    # ":memory:" = 디스크에 아무것도 남기지 않는 Qdrant
    return VectorStore(QdrantClient(":memory:"), "test_reviews")

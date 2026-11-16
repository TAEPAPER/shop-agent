"""Qdrant 벡터 저장소 래퍼. Spring의 Repository 계층에 해당한다."""

import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from shop_agent.config import Settings
from shop_agent.models import Chunk, SearchHit


def create_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)  # Docker 서버 모드
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.qdrant_path))  # 로컬 파일 모드


def _point_id(chunk_id: str) -> str:
    # Qdrant의 point id는 정수나 UUID만 허용한다. 같은 chunk_id는 항상 같은 UUID가 되도록 uuid5를 쓴다.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str):
        self.client = client
        self.collection = collection

    def recreate(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        points = [
            PointStruct(
                id=_point_id(c.chunk_id),
                vector=v.tolist(),
                payload={**c.payload, "chunk_id": c.chunk_id, "text": c.text},
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points=points)

    def count(self) -> int:
        return self.client.count(self.collection).count

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        result = self.client.query_points(
            self.collection, query=query_vector.tolist(), limit=top_k, with_payload=True
        )
        hits = []
        for p in result.points:
            payload = dict(p.payload or {})
            hits.append(
                SearchHit(
                    chunk_id=payload.pop("chunk_id"),
                    text=payload.pop("text"),
                    score=p.score,
                    payload=payload,
                )
            )
        return hits

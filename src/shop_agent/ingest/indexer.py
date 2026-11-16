"""청크를 임베딩해서 벡터 저장소에 적재한다."""

from shop_agent.models import Chunk
from shop_agent.retrieval.embedder import Embedder
from shop_agent.retrieval.store import VectorStore


def index_chunks(chunks: list[Chunk], embedder: Embedder, store: VectorStore) -> int:
    if not chunks:
        raise ValueError("적재할 청크가 없습니다.")
    vectors = embedder.embed([c.text for c in chunks])
    # 1주차: 매번 전체 재적재. 증분 적재는 데이터가 커지면 도입
    store.recreate(dim=vectors.shape[1])
    store.upsert(chunks, vectors)
    return store.count()

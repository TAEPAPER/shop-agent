from shop_agent.ingest.indexer import index_chunks
from shop_agent.pipeline import RagPipeline


def test_index_stores_every_chunk(chunks, embedder, store):
    assert index_chunks(chunks, embedder, store) == len(chunks)


def test_reindex_does_not_duplicate(chunks, embedder, store):
    index_chunks(chunks, embedder, store)
    assert index_chunks(chunks, embedder, store) == len(chunks)


def test_retrieve_returns_top_k_sorted_with_payload(chunks, embedder, store):
    index_chunks(chunks, embedder, store)
    pipeline = RagPipeline(embedder, store, top_k=3)

    hits = pipeline.retrieve("에어프라이어 바스켓 세척")

    assert len(hits) == 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert hits[0].payload["product_id"] == "P002"
    assert "review_text" in hits[0].payload

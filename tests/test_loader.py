from shop_agent.ingest.loader import build_chunks
from shop_agent.models import Product, Review


def test_sample_data_builds_one_chunk_per_review(chunks):
    assert len(chunks) == 32
    assert len({c.chunk_id for c in chunks}) == 32  # chunk_id 중복 없음


def test_chunk_text_includes_product_context(chunks):
    first = next(c for c in chunks if c.chunk_id == "R0001")
    assert "사운드핏 무선 이어폰 S2" in first.text
    assert first.payload["product_id"] == "P001"
    assert first.payload["rating"] == 5


def test_review_without_product_is_skipped():
    products = {
        "P1": Product(product_id="P1", name="a", category="c", brand="b", price=1, description="d")
    }
    reviews = [
        Review(review_id="R1", product_id="P1", rating=5, text="좋아요"),
        Review(review_id="R2", product_id="P404", rating=1, text="상품 없음"),
    ]
    assert [c.chunk_id for c in build_chunks(products, reviews)] == ["R1"]

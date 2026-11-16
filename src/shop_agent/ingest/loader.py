"""원본 데이터(JSONL)를 읽어 검색 단위(Chunk)로 바꾼다."""

import json
from pathlib import Path

from shop_agent.models import Chunk, Product, Review


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_products(path: Path) -> dict[str, Product]:
    return {p.product_id: p for p in (Product(**row) for row in _read_jsonl(path))}


def load_reviews(path: Path) -> list[Review]:
    return [Review(**row) for row in _read_jsonl(path)]


def build_chunks(products: dict[str, Product], reviews: list[Review]) -> list[Chunk]:
    """리뷰마다 청크 1개를 만든다.

    리뷰 본문만 임베딩하면 "에어프라이어 소음"처럼 상품명이 들어간 질문과 잘 매칭되지 않는다.
    그래서 카테고리와 상품명을 본문 앞에 붙여 함께 임베딩한다. (2주차에 이 선택의 효과를 수치로 비교해 볼 것)
    """
    chunks = []
    for r in reviews:
        p = products.get(r.product_id)
        if p is None:
            continue  # 상품 정보가 없는 리뷰는 버린다
        text = f"[{p.category}] {p.name} | 평점 {r.rating}점\n{r.text}"
        chunks.append(
            Chunk(
                chunk_id=r.review_id,
                text=text,
                payload={
                    "review_id": r.review_id,
                    "product_id": p.product_id,
                    "product_name": p.name,
                    "category": p.category,
                    "price": p.price,
                    "rating": r.rating,
                    "review_text": r.text,
                },
            )
        )
    return chunks

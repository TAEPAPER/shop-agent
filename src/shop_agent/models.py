"""도메인 모델. Spring의 DTO/엔티티에 해당한다.

Pydantic 모델은 생성 시점에 타입을 검증하므로, 잘못된 데이터가 파이프라인 깊숙이 들어가기 전에 막아준다.
"""

from pydantic import BaseModel


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    brand: str
    price: int
    description: str


class Review(BaseModel):
    review_id: str
    product_id: str
    rating: int
    text: str


class Chunk(BaseModel):
    """검색 단위. 지금은 리뷰 1개 = 청크 1개.

    text: 임베딩할 문자열 (상품 정보 + 리뷰 본문)
    payload: 검색 후 답변 생성·필터링에 쓸 메타데이터
    """

    chunk_id: str
    text: str
    payload: dict


class SearchHit(BaseModel):
    chunk_id: str
    score: float
    text: str
    payload: dict

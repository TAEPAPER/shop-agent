from shop_agent.llm.generator import build_user_message
from shop_agent.models import SearchHit


def _hit(i: int) -> SearchHit:
    return SearchHit(
        chunk_id=f"R{i}",
        score=0.9,
        text="",
        payload={
            "product_name": f"상품{i}",
            "category": "생활용품",
            "price": 25000,
            "rating": 4,
            "review_text": f"리뷰 본문 {i}",
        },
    )


def test_reviews_are_numbered_in_order():
    msg = build_user_message("보냉 잘 돼요?", [_hit(1), _hit(2)])
    assert msg.index("[1] 상품: 상품1") < msg.index("[2] 상품: 상품2")
    assert "25,000원" in msg


def test_question_comes_after_reviews():
    msg = build_user_message("보냉 잘 돼요?", [_hit(1)])
    assert msg.index("</reviews>") < msg.index("질문: 보냉 잘 돼요?")

"""검색된 리뷰를 근거로 Claude가 답변을 생성한다."""

from dataclasses import dataclass

import anthropic

from shop_agent.models import SearchHit

SYSTEM_PROMPT = """당신은 온라인 쇼핑몰의 상품 상담 어시스턴트입니다.
사용자 질문에 <reviews> 안의 고객 리뷰만 근거로 한국어로 답하세요.

- 근거로 쓴 리뷰 번호를 문장 끝에 [1], [2]처럼 표기하세요.
- 리뷰에 없는 내용은 추측하지 말고 "리뷰에서 확인할 수 없습니다"라고 말하세요.
- 장점과 단점이 함께 있으면 둘 다 알려주세요.
- 3~6문장으로 간결하게 답하세요."""


def build_user_message(question: str, hits: list[SearchHit]) -> str:
    """검색 결과를 번호 붙인 리뷰 목록으로 만들어 질문 앞에 둔다."""
    lines = ["<reviews>"]
    for i, h in enumerate(hits, start=1):
        p = h.payload
        lines.append(
            f"[{i}] 상품: {p['product_name']} ({p['category']}, {p['price']:,}원) | "
            f"평점 {p['rating']}점\n{p['review_text']}"
        )
    lines.append("</reviews>")
    lines.append(f"\n질문: {question}")
    return "\n".join(lines)


@dataclass
class Answer:
    text: str
    model: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int


class AnswerGenerator:
    def __init__(
        self, model: str, effort: str = "low", max_tokens: int = 4000, api_key: str | None = None
    ):
        # api_key가 None이면 SDK가 환경변수 ANTHROPIC_API_KEY를 찾는다. 코드에 키를 직접 쓰지 않는다.
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def generate(self, question: str, hits: list[SearchHit]) -> Answer:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_message(question, hits)}],
            output_config={"effort": self.effort},
            # 안전 분류기가 요청을 거절하면 서버가 다른 모델로 자동 재시도한다.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            text = "죄송합니다. 이 질문에는 답변할 수 없습니다."
        else:
            # content에는 thinking 블록 등도 섞여 있으므로 text 블록만 모은다.
            text = "".join(b.text for b in response.content if b.type == "text")
        return Answer(
            text=text,
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

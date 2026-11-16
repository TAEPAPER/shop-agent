"""RAG 파이프라인: 질문 → 검색 → 답변 생성. Spring의 Service 계층에 해당한다."""

from dataclasses import dataclass

from shop_agent.llm.generator import Answer, AnswerGenerator
from shop_agent.models import SearchHit
from shop_agent.retrieval.embedder import Embedder
from shop_agent.retrieval.store import VectorStore


@dataclass
class RagResult:
    question: str
    hits: list[SearchHit]
    answer: Answer | None


class RagPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        generator: AnswerGenerator | None = None,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.store = store
        self.generator = generator
        self.top_k = top_k

    def retrieve(self, question: str) -> list[SearchHit]:
        query_vector = self.embedder.embed([question])[0]
        return self.store.search(query_vector, self.top_k)

    def ask(self, question: str) -> RagResult:
        hits = self.retrieve(question)
        if self.generator is None:
            raise RuntimeError("답변 생성기가 설정되지 않았습니다.")
        return RagResult(question, hits, self.generator.generate(question, hits))

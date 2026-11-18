"""평가 실행기: 정답셋의 질문마다 검색 → 지표 계산 → 평균. 통합 테스트 러너에 해당한다.

평균만 보면 "무엇이 바뀌었는지" 알 수 없으므로 문항별 결과와 유형별 평균을 함께 남긴다.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Protocol

from shop_agent.eval.golden import GoldenItem
from shop_agent.eval.metrics import hit_at_k, recall_at_k, reciprocal_rank
from shop_agent.models import SearchHit


class Retriever(Protocol):
    """retrieve()만 있으면 된다. RagPipeline이든, 4강의 BM25든 같은 실행기로 잰다."""

    def retrieve(self, question: str) -> list[SearchHit]: ...


@dataclass
class QuestionResult:
    qid: str
    question: str
    type: str
    relevant_ids: list[str]
    retrieved_ids: list[str]
    recall: float
    rr: float
    hit: float

    @property
    def missed_ids(self) -> list[str]:
        return [rid for rid in self.relevant_ids if rid not in self.retrieved_ids]


@dataclass
class Summary:
    count: int
    recall: float
    mrr: float
    hit: float


@dataclass
class EvalReport:
    k: int
    overall: Summary
    by_type: dict[str, Summary]
    results: list[QuestionResult]
    unanswerable_qids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for r, src in zip(d["results"], self.results, strict=True):
            r["missed_ids"] = src.missed_ids
        return d


def _summarize(results: list[QuestionResult]) -> Summary:
    if not results:
        return Summary(0, 0.0, 0.0, 0.0)
    return Summary(
        count=len(results),
        recall=mean(r.recall for r in results),
        mrr=mean(r.rr for r in results),
        hit=mean(r.hit for r in results),
    )


def evaluate(retriever: Retriever, items: list[GoldenItem], k: int) -> EvalReport:
    results: list[QuestionResult] = []
    unanswerable: list[str] = []
    for item in items:
        if not item.answerable:
            unanswerable.append(item.qid)
            continue
        hits = retriever.retrieve(item.question)
        retrieved = [h.payload["review_id"] for h in hits][:k]
        results.append(
            QuestionResult(
                qid=item.qid,
                question=item.question,
                type=item.type,
                relevant_ids=item.relevant_ids,
                retrieved_ids=retrieved,
                recall=recall_at_k(retrieved, item.relevant_ids, k),
                rr=reciprocal_rank(retrieved, item.relevant_ids, k),
                hit=hit_at_k(retrieved, item.relevant_ids, k),
            )
        )

    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        groups[r.type or "(미지정)"].append(r)

    return EvalReport(
        k=k,
        overall=_summarize(results),
        by_type={t: _summarize(rs) for t, rs in sorted(groups.items())},
        results=results,
        unanswerable_qids=unanswerable,
    )

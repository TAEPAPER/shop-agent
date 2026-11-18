import pytest

from shop_agent.eval.golden import GoldenItem
from shop_agent.eval.runner import evaluate
from shop_agent.models import SearchHit


class FakeRetriever:
    """질문마다 정해 둔 id 순서대로 hit을 돌려준다."""

    def __init__(self, answers: dict[str, list[str]]):
        self.answers = answers

    def retrieve(self, question: str) -> list[SearchHit]:
        return [
            SearchHit(chunk_id=rid, score=1.0 - i * 0.1, text="", payload={"review_id": rid})
            for i, rid in enumerate(self.answers[question])
        ]


ITEMS = [
    GoldenItem(qid="Q1", question="a", relevant_ids=["A", "B"], type="여러정답"),
    GoldenItem(qid="Q2", question="b", relevant_ids=["C"], type="상품명"),
    GoldenItem(qid="Q3", question="c", relevant_ids=["D"], type="상품명"),
    GoldenItem(qid="Q4", question="d", relevant_ids=[], type="답없음"),
]
RETRIEVER = FakeRetriever(
    {
        "a": ["X", "A", "Y", "Z", "W"],  # recall 0.5, RR 0.5
        "b": ["C", "X", "Y", "Z", "W"],  # recall 1.0, RR 1.0
        "c": ["X", "Y", "Z", "W", "V", "D"],  # D는 6위 → 0
    }
)


def test_overall_averages_skip_unanswerable():
    report = evaluate(RETRIEVER, ITEMS, k=5)
    assert report.overall.count == 3
    assert report.overall.recall == 0.5
    assert report.overall.mrr == 0.5
    assert report.overall.hit == pytest.approx(2 / 3)
    assert report.unanswerable_qids == ["Q4"]


def test_by_type_and_per_question_detail():
    report = evaluate(RETRIEVER, ITEMS, k=5)
    assert report.by_type["상품명"].count == 2
    assert report.by_type["상품명"].recall == 0.5
    assert report.by_type["여러정답"].recall == 0.5

    q1 = next(r for r in report.results if r.qid == "Q1")
    assert q1.missed_ids == ["B"]
    q3 = next(r for r in report.results if r.qid == "Q3")
    assert q3.retrieved_ids == ["X", "Y", "Z", "W", "V"]  # top-k로 잘린다


def test_report_serializes_with_missed_ids():
    d = evaluate(RETRIEVER, ITEMS, k=5).to_dict()
    assert d["results"][0]["missed_ids"] == ["B"]
    assert d["overall"]["count"] == 3

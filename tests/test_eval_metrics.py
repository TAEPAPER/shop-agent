"""3강 4절의 손계산 예시를 그대로 테스트 케이스로 쓴다."""

from statistics import mean

import pytest

from shop_agent.eval.metrics import hit_at_k, recall_at_k, reciprocal_rank

# (정답, top-5 결과) — 4절 표의 질문 가·나·다
GA = ({"A", "B"}, ["X", "A", "Y", "Z", "W"])
NA = ({"C"}, ["C", "X", "Y", "Z", "W"])
DA = ({"D", "E"}, ["X", "Y", "Z", "W", "V", "U", "D"])  # D는 7위


@pytest.mark.parametrize(
    ("case", "recall", "rr", "hit"),
    [(GA, 0.5, 0.5, 1.0), (NA, 1.0, 1.0, 1.0), (DA, 0.0, 0.0, 0.0)],
)
def test_lecture_examples(case, recall, rr, hit):
    relevant, retrieved = case
    assert recall_at_k(retrieved, relevant, 5) == recall
    assert reciprocal_rank(retrieved, relevant, 5) == rr
    assert hit_at_k(retrieved, relevant, 5) == hit


def test_lecture_averages():
    cases = [GA, NA, DA]
    assert mean(recall_at_k(r, g, 5) for g, r in cases) == 0.5
    assert mean(reciprocal_rank(r, g, 5) for g, r in cases) == 0.5
    assert mean(hit_at_k(r, g, 5) for g, r in cases) == pytest.approx(2 / 3)


def test_outside_top_k_counts_as_missing():
    # D가 7위면 top-5에서는 없는 것과 같지만, top-7에서는 찾은 것이다
    relevant, retrieved = DA
    assert recall_at_k(retrieved, relevant, 7) == 0.5
    assert reciprocal_rank(retrieved, relevant, 7) == pytest.approx(1 / 7)


def test_multiple_answers_partial_recall():
    # 4절 '한눈에 보기': 정답 A,B,C 중 A(2등)와 C(4등)만 찾음
    retrieved = ["X", "A", "Y", "C", "Z"]
    assert recall_at_k(retrieved, {"A", "B", "C"}, 5) == pytest.approx(2 / 3)
    assert reciprocal_rank(retrieved, {"A", "B", "C"}, 5) == 0.5


def test_fewer_results_than_k():
    assert recall_at_k(["A"], {"A"}, 5) == 1.0


@pytest.mark.parametrize("fn", [recall_at_k, reciprocal_rank, hit_at_k])
def test_empty_relevant_is_rejected(fn):
    with pytest.raises(ValueError):
        fn(["A"], set(), 5)


@pytest.mark.parametrize("fn", [recall_at_k, reciprocal_rank, hit_at_k])
def test_invalid_k_is_rejected(fn):
    with pytest.raises(ValueError):
        fn(["A"], {"A"}, 0)

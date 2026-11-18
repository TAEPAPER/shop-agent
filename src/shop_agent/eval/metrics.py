"""검색 지표. 입력이 같으면 출력이 같은 순수 함수라서 평범하게 단위 테스트한다.

retrieved: 검색 결과 id 목록 (1등부터 순서대로)
relevant:  정답 id 집합 (순서 없음)

정답이 없는 문항(relevant가 비어 있음)은 검색 지표로 잴 수 없으므로 ValueError를 던진다.
그런 문항은 평가 실행기에서 미리 걸러낸다.
"""

from collections.abc import Collection, Sequence


def _check(relevant: Collection[str], k: int) -> None:
    if not relevant:
        raise ValueError("정답이 없는 문항은 검색 지표로 잴 수 없습니다.")
    if k < 1:
        raise ValueError(f"k는 1 이상이어야 합니다: {k}")


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """정답 중 top-k 안에 들어온 비율. 순서는 보지 않는다."""
    _check(relevant, k)
    found = set(retrieved[:k]) & set(relevant)
    return len(found) / len(set(relevant))


def reciprocal_rank(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1 / 첫 정답의 순위. top-k 안에 정답이 없으면 0."""
    _check(relevant, k)
    relevant = set(relevant)
    for rank, rid in enumerate(retrieved[:k], start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def hit_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """정답이 하나라도 top-k 안에 있으면 1, 없으면 0."""
    _check(relevant, k)
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0

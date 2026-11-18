"""정답셋(golden set) 읽기.

정답셋은 테스트 픽스처다. 픽스처에 오타가 있으면 점수가 조용히 틀어지므로,
없는 리뷰 id나 중복 qid는 읽는 시점에 바로 에러로 막는다.
"""

import json
from collections.abc import Collection
from pathlib import Path

from pydantic import BaseModel


class GoldenItem(BaseModel):
    qid: str
    question: str
    relevant_ids: list[str]  # 비어 있으면 '답이 없는 질문' (검색 지표에서 제외)
    type: str = ""  # 3강 3절 표의 질문 유형 (상품명, 다른단어, 헷갈리는상품 ...)
    note: str = ""  # 정답을 고른 기준

    @property
    def answerable(self) -> bool:
        return bool(self.relevant_ids)


def load_golden(path: Path, known_review_ids: Collection[str]) -> list[GoldenItem]:
    known = set(known_review_ids)
    items: list[GoldenItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = GoldenItem(**json.loads(line))
            if item.qid in seen:
                raise ValueError(f"{path}:{lineno}: qid가 중복됩니다: {item.qid}")
            unknown = [rid for rid in item.relevant_ids if rid not in known]
            if unknown:
                raise ValueError(
                    f"{path}:{lineno}: 없는 리뷰 id가 있습니다 ({item.qid}): {unknown}"
                )
            seen.add(item.qid)
            items.append(item)
    return items

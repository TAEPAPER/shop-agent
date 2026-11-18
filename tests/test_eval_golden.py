import json

import pytest

from shop_agent.eval.golden import load_golden

KNOWN = {"R0001", "R0002", "R0003"}


def _write(tmp_path, rows):
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


def test_loads_items_and_flags_unanswerable(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "qid": "Q001",
                "question": "이어폰 단점",
                "relevant_ids": ["R0002", "R0003"],
                "type": "여러정답",
            },
            {"qid": "Q002", "question": "이어폰 색상은?", "relevant_ids": [], "type": "답없음"},
        ],
    )
    items = load_golden(path, KNOWN)
    assert [i.qid for i in items] == ["Q001", "Q002"]
    assert items[0].answerable and not items[1].answerable


def test_rejects_unknown_review_id(tmp_path):
    path = _write(tmp_path, [{"qid": "Q001", "question": "q", "relevant_ids": ["R9999"]}])
    with pytest.raises(ValueError, match="R9999"):
        load_golden(path, KNOWN)


def test_rejects_duplicate_qid(tmp_path):
    row = {"qid": "Q001", "question": "q", "relevant_ids": ["R0001"]}
    path = _write(tmp_path, [row, row])
    with pytest.raises(ValueError, match="중복"):
        load_golden(path, KNOWN)

"""명령줄 진입점.

uv run shop-agent ingest           샘플 데이터를 임베딩해서 적재
uv run shop-agent search "질문"     검색 결과만 보기 (API 키 불필요)
uv run shop-agent ask "질문"        검색 + 답변 생성
uv run shop-agent eval             정답셋으로 검색 평가 (API 키 불필요)
"""

import argparse
import sys
import time

from rich.console import Console
from rich.table import Table

from shop_agent.config import get_settings

console = Console()


def _embedder(settings):
    from shop_agent.retrieval.embedder import SentenceTransformerEmbedder

    with console.status(f"임베딩 모델 로딩 중: {settings.embed_model}"):
        emb = SentenceTransformerEmbedder(
            settings.embed_model, settings.embed_device, settings.embed_batch_size
        )
    console.print(f"[dim]임베딩 장치: {emb.device}[/dim]")
    return emb


def _store(settings):
    from shop_agent.retrieval.store import VectorStore, create_client

    return VectorStore(create_client(settings), settings.collection)


def cmd_ingest(settings) -> None:
    from shop_agent.ingest.indexer import index_chunks
    from shop_agent.ingest.loader import build_chunks, load_products, load_reviews

    products = load_products(settings.products_path)
    reviews = load_reviews(settings.reviews_path)
    chunks = build_chunks(products, reviews)
    console.print(f"상품 {len(products)}개, 리뷰 {len(reviews)}개 → 청크 {len(chunks)}개")

    embedder = _embedder(settings)
    started = time.perf_counter()
    count = index_chunks(chunks, embedder, _store(settings))
    console.print(f"[green]적재 완료[/green]: {count}건 ({time.perf_counter() - started:.1f}초)")


def _print_hits(hits) -> None:
    table = Table(title="검색 결과")
    table.add_column("#", justify="right")
    table.add_column("점수", justify="right")
    table.add_column("상품")
    table.add_column("리뷰")
    for i, h in enumerate(hits, start=1):
        table.add_row(str(i), f"{h.score:.3f}", h.payload["product_name"], h.payload["review_text"])
    console.print(table)


def cmd_search(settings, question: str) -> None:
    from shop_agent.pipeline import RagPipeline

    pipeline = RagPipeline(_embedder(settings), _store(settings), top_k=settings.top_k)
    _print_hits(pipeline.retrieve(question))


def cmd_ask(settings, question: str) -> None:
    from shop_agent.llm.generator import AnswerGenerator
    from shop_agent.pipeline import RagPipeline

    key = settings.anthropic_api_key
    generator = AnswerGenerator(
        settings.llm_model,
        settings.llm_effort,
        settings.llm_max_tokens,
        api_key=key.get_secret_value() if key else None,
    )
    pipeline = RagPipeline(_embedder(settings), _store(settings), generator, settings.top_k)

    started = time.perf_counter()
    result = pipeline.ask(question)
    elapsed = time.perf_counter() - started

    _print_hits(result.hits)
    a = result.answer
    console.print(f"\n[bold]답변[/bold]\n{a.text}\n")
    console.print(
        f"[dim]모델 {a.model} | 입력 {a.input_tokens} / 출력 {a.output_tokens} 토큰 | "
        f"{elapsed:.1f}초[/dim]"
    )


def cmd_eval(settings, k: int | None, golden: str | None, out: str | None) -> None:
    import json
    from pathlib import Path

    from shop_agent.eval.golden import load_golden
    from shop_agent.eval.runner import evaluate
    from shop_agent.ingest.loader import load_reviews
    from shop_agent.pipeline import RagPipeline

    k = k or settings.top_k
    golden_path = Path(golden) if golden else settings.golden_path
    review_ids = {r.review_id for r in load_reviews(settings.reviews_path)}
    items = load_golden(golden_path, review_ids)

    pipeline = RagPipeline(_embedder(settings), _store(settings), top_k=k)
    report = evaluate(pipeline, items, k)

    o = report.overall
    console.print(
        f"\n[bold]{golden_path}[/bold]  문항 {len(items)}개 "
        f"(채점 {o.count}개, 답없음 {len(report.unanswerable_qids)}개 제외)\n"
        f"[bold]recall@{k} {o.recall:.3f} | MRR {o.mrr:.3f} | hit@{k} {o.hit:.3f}[/bold]"
    )

    by_type = Table(title="유형별")
    for col in ("유형", "문항", f"recall@{k}", "MRR", f"hit@{k}"):
        by_type.add_column(col, justify="left" if col == "유형" else "right")
    for t, s in report.by_type.items():
        by_type.add_row(t, str(s.count), f"{s.recall:.3f}", f"{s.mrr:.3f}", f"{s.hit:.3f}")
    console.print(by_type)

    per_q = Table(title="문항별 (recall < 1 은 노란색)")
    for col in ("qid", "유형", "질문", f"top-{k}", "놓친 정답", "recall", "RR"):
        per_q.add_column(col)
    for r in report.results:
        style = "yellow" if r.recall < 1 else None
        per_q.add_row(
            r.qid,
            r.type,
            r.question,
            ", ".join(r.retrieved_ids),
            ", ".join(r.missed_ids),
            f"{r.recall:.2f}",
            f"{r.rr:.2f}",
            style=style,
        )
    console.print(per_q)

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        console.print(f"[green]저장[/green]: {out_path}")


def main() -> None:
    # Windows 콘솔에서 한글이 깨지지 않도록 출력 인코딩을 UTF-8로 맞춘다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="shop-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="샘플 데이터를 임베딩해서 적재")
    sub.add_parser("search", help="검색 결과만 보기").add_argument("question")
    sub.add_parser("ask", help="검색 + 답변 생성").add_argument("question")
    ev = sub.add_parser("eval", help="정답셋으로 검색 평가 (recall@k, MRR)")
    ev.add_argument("--k", type=int, help="top-k (기본: SHOP_TOP_K)")
    ev.add_argument("--golden", help="정답셋 경로 (기본: data/eval/golden.jsonl)")
    ev.add_argument("--out", help="리포트를 JSON으로 저장할 경로")
    args = parser.parse_args()

    settings = get_settings()
    if args.command == "ingest":
        cmd_ingest(settings)
    elif args.command == "search":
        cmd_search(settings, args.question)
    elif args.command == "ask":
        cmd_ask(settings, args.question)
    elif args.command == "eval":
        cmd_eval(settings, args.k, args.golden, args.out)


if __name__ == "__main__":
    main()

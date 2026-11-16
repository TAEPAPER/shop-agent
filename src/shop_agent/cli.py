"""명령줄 진입점.

uv run shop-agent ingest           샘플 데이터를 임베딩해서 적재
uv run shop-agent search "질문"     검색 결과만 보기 (API 키 불필요)
uv run shop-agent ask "질문"        검색 + 답변 생성
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

    generator = AnswerGenerator(settings.llm_model, settings.llm_effort, settings.llm_max_tokens)
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


def main() -> None:
    # Windows 콘솔에서 한글이 깨지지 않도록 출력 인코딩을 UTF-8로 맞춘다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="shop-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="샘플 데이터를 임베딩해서 적재")
    sub.add_parser("search", help="검색 결과만 보기").add_argument("question")
    sub.add_parser("ask", help="검색 + 답변 생성").add_argument("question")
    args = parser.parse_args()

    settings = get_settings()
    if args.command == "ingest":
        cmd_ingest(settings)
    elif args.command == "search":
        cmd_search(settings, args.question)
    elif args.command == "ask":
        cmd_ask(settings, args.question)


if __name__ == "__main__":
    main()

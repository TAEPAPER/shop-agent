"""애플리케이션 설정.

Spring의 application.yml + @ConfigurationProperties 역할이다.
환경변수(또는 .env)에서 SHOP_ 접두사가 붙은 값을 읽어 타입이 있는 객체로 만든다.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SHOP_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    # 이 값만 SHOP_ 접두사 없이 표준 이름(ANTHROPIC_API_KEY)으로 읽는다.
    # SecretStr: print나 로그에 찍혀도 '**********'로 가려진다.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-sonnet-5-5"
    llm_effort: str = "low"  # 검색 결과 기반 짧은 답변이라 low로 충분하다
    llm_max_tokens: int = 4000

    # 임베딩
    embed_model: str = "BAAI/bge-m3"
    embed_device: str = "auto"  # auto | cuda | cpu
    embed_batch_size: int = 16

    # 벡터 DB
    qdrant_url: str | None = None  # 지정하면 서버 모드(Docker)
    qdrant_path: Path = Path("data/qdrant")  # 지정 안 하면 로컬 파일 모드
    collection: str = "reviews"

    # 데이터
    products_path: Path = Path("data/sample/products.jsonl")
    reviews_path: Path = Path("data/sample/reviews.jsonl")

    # 검색
    top_k: int = 5

    # 평가 (3강)
    golden_path: Path = Path("data/eval/golden.jsonl")


@lru_cache
def get_settings() -> Settings:
    return Settings()

# shop-agent

상품 정보와 고객 리뷰를 근거로 답하는 **한국어 쇼핑 어시스턴트 에이전트**입니다.
"RAG 챗봇을 만들었다"에서 끝나지 않고, **평가 체계를 세우고 수치로 개선을 증명하는 것**을 목표로 합니다.

> 진행 상태: 1주차 (기준선 RAG)

## 구조

```
질문 → [임베딩 bge-m3] → [Qdrant 벡터 검색 top-k] → [Claude: 리뷰 근거로 답변 + 출처 표기]
```

| 영역 | 기술 |
|---|---|
| 언어·패키지 | Python 3.12, uv |
| LLM | Claude API (기본 `claude-sonnet-5-5`) |
| 임베딩 | BAAI/bge-m3 (로컬, GPU 자동 사용) |
| 벡터 DB | Qdrant (로컬 파일 모드, Docker 서버 모드 전환 가능) |

## 실행

```bash
uv sync
cp .env.example .env        # ANTHROPIC_API_KEY 입력
uv run python -m shop_agent ingest    # 샘플 데이터 임베딩 → Qdrant 적재
uv run python -m shop_agent search "소음이 적은 에어프라이어"   # 검색만 (API 키 불필요)
uv run python -m shop_agent ask "노이즈 캔슬링 좋은 이어폰 있어?"  # 검색 + 답변 생성
uv run python -m pytest
```

> Windows 스마트 앱 컨트롤이 켜진 PC에서는 uv가 만드는 실행 파일(`shop-agent.exe`, `pytest.exe`)이 차단될 수 있어서
> `python -m` 형태로 실행합니다.

## 데이터

`data/sample/`은 파이프라인 개발용으로 직접 작성한 **가상 데이터**(상품 8개, 리뷰 32개)입니다.
실제 데이터셋은 이후 교체합니다.

## 로드맵

- [x] 1주차: 기준선 RAG (dense 검색)
- [ ] 1주차: 정답셋 50개 + 평가 스크립트 → 기준선 수치
- [ ] 2주차: Kiwi BM25 + 하이브리드(RRF) + 리랭커
- [ ] 3주차: 질의 재작성, 메타데이터 필터, 에이전트 도구 호출
- [ ] 4주차: FastAPI 스트리밍, 캐시, 트레이싱, 지연시간·비용 측정
- [ ] 5주차: CI 평가 회귀 검사, 데모 UI

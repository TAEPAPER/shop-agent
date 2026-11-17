import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from shop_agent.ingest.loader import load_products, load_reviews, build_chunks

# ① 데이터 읽기
products = load_products(Path("data/sample/products.jsonl"))
reviews = load_reviews(Path("data/sample/reviews.jsonl"))

# ② 청크 만들기: 이미 있는 함수를 호출만 한다
chunks = build_chunks(products, reviews)
print(len(chunks))  # 32가 나와야 함

# ③ 청크 임베딩: Chunk 객체가 아니라 임베딩할 문자열(text) 목록을 넘긴다
model = SentenceTransformer("BAAI/bge-m3", device="cuda")
texts = [c.text for c in chunks]  # 자바의 chunks.stream().map(Chunk::getText).toList()
doc_vecs = model.encode(texts, normalize_embeddings=True)
print(doc_vecs.shape)  # (32, 1024): 청크 32개 × 숫자 1024개

# ④ 질문 임베딩: 문자열 1개를 넘기면 모양이 (1024,)
query = "밤에 써도 조용한 에어프라이어"
q_vec = model.encode(query, normalize_embeddings=True)

# ⑤ 전수 비교: 내적 → 정렬 → top-5
# 정규화했으므로 나눗셈 없이 내적만 하면 코사인 유사도다.
# (32, 1024) @ (1024,) → (32,): 32개 청크의 점수를 한 번에 계산
scores = doc_vecs @ q_vec

top5 = np.argsort(-scores)[:5]  # 점수 내림차순 정렬 후 앞 5개의 위치
for rank, i in enumerate(top5, 1):
    print(rank, f"{scores[i]:.3f}", chunks[i].chunk_id, chunks[i].text[:40])

"""문장을 벡터로 바꾸는 임베더.

Protocol은 Java의 interface와 같다. 테스트에서는 모델을 내려받지 않는 가짜 임베더로 바꿔 끼운다.
"""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """texts -> (len(texts), dim) 크기의 정규화된 벡터"""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 16):
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True: 벡터 길이를 1로 맞춘다. 그러면 내적 = 코사인 유사도가 된다.
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )

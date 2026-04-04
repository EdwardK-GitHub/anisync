from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.settings import settings


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    """
    Lazily load the embedding model once per process.

    We use the same model for:
    - offline anime embeddings
    - runtime query embeddings
    """
    return SentenceTransformer(settings.embedding_model)


def embed_text(text: str) -> list[float]:
    """
    Convert text into a normalized embedding vector.

    We return a plain Python list so it can be stored easily in pgvector columns.
    """
    model = get_embedding_model()
    vector = model.encode([text], normalize_embeddings=True)[0]
    vector = np.asarray(vector, dtype=np.float32)
    return vector.tolist()
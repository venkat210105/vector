"""Turns text into vectors using a small local model (no API key needed).

BAAI/bge-small-en-v1.5 gives 384-number vectors. The `dim` of the collection
must equal this model's output size, which is why DIM is exported here.
"""

import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384


class Embedder:
    def __init__(self) -> None:
        self._model = TextEmbedding(MODEL_NAME)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Vectors for the chunks that get stored. One row per text."""
        return np.array(list(self._model.passage_embed(texts)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Vector for a question. BGE models embed queries slightly differently
        from passages, so this uses the query path."""
        return np.array(next(iter(self._model.query_embed(text))), dtype=np.float32)

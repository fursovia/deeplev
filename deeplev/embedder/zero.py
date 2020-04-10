import numpy as np

from deeplev.embedder.abstract import IEmbedder, Embedding


class ZeroEmbedder(IEmbedder):
    def __init__(self, dimension: int):
        self._dim = dimension

    def embed(self, text: str) -> Embedding:
        return np.zeros(self._dim)

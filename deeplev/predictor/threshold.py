from typing import Sequence, Callable

from scipy.spatial.distance import euclidean

from deeplev.predictor.abstract import IPredictor
from deeplev.embedder import EmbedderLike, Embedding, load_embedder


class ThresholdPredictor(IPredictor):
    def __init__(
            self,
            embedder_like: EmbedderLike,
            distance_threshold: float = 0,
            similarity_func: Callable[[Embedding, Embedding], float] = euclidean,
    ):
        self._embedder = load_embedder(embedder_like)
        self._distance_threshold = distance_threshold
        self._similarity_func = similarity_func

        self._fitted_texts = None
        self._fitted_embeddings = None

    def fit(self, texts: Sequence[str]):
        self._fitted_texts = texts
        self._fitted_embeddings = [self._embedder.embed(text) for text in texts]

    def get_similar_texts(self, text: str) -> Sequence[str]:
        text_embedding = self._embedder.embed(text)

        return [
            self._fitted_texts[number]
            for number, fitted_embedding in enumerate(self._fitted_embeddings)
            if self._similarity_func(text_embedding, fitted_embedding) <= self._distance_threshold
        ]

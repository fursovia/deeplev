from typing import List, Callable, Sequence, Optional

import numpy as np
from allennlp.predictors.predictor import Predictor

from deeplev.model import DeepLevenshtein
from deeplev.utils import clean_sequence
from deeplev.knn import ApproxKNN
from deeplev.predictors import EmbedderPredictor


@Predictor.register("searcher")
class SearchPredictor(EmbedderPredictor):
    def __init__(
        self, model: DeepLevenshtein, num_neighbors: int = 10, preprocessor: Callable[[str], str] = clean_sequence,
    ) -> None:
        super().__init__(model=model, preprocessor=preprocessor)
        self._num_neighbors = num_neighbors
        self._knn = ApproxKNN(n_neighbors=num_neighbors)
        self._data: np.ndarray = None

    def fit(self, data: Sequence[str]) -> "SearchPredictor":
        embeddings = []
        for seq in data:
            embeddings.append(self.get_embeddings(seq))
        embeddings = np.array(embeddings)
        self._data = np.array(data)
        self._knn.fit(embeddings)
        return self

    def find_neighbors(self, data: Sequence[str], n_neighbors: Optional[int] = None) -> List[List[str]]:
        n_neighbors = n_neighbors or self._num_neighbors
        embeddings = []
        for seq in data:
            embeddings.append(self.get_embeddings(seq))
        embeddings = np.array(embeddings)
        _, indexes = self._knn.kneighbors(embeddings, n_neighbors=n_neighbors)
        return self._data[indexes].tolist()

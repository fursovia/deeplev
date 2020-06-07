from typing import List, Sequence, Optional

import numpy as np
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model
from allennlp.data import DatasetReader

from deeplev.modules.knn import ApproxKNN
from deeplev.predictors import EmbedderPredictor


@Predictor.register("searcher")
class SearchPredictor(EmbedderPredictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader, num_neighbors: int = 3) -> None:
        super().__init__(model=model, dataset_reader=dataset_reader)
        self._num_neighbors = num_neighbors
        self._knn = ApproxKNN(n_neighbors=num_neighbors)
        self._data: np.ndarray = None

    def fit(self, data: Sequence[str]) -> "SearchPredictor":
        embeddings = self.get_embeddings(data)
        self._data = np.array(data)
        self._knn.fit(embeddings)
        return self

    def find_neighbors(self, data: Sequence[str], n_neighbors: Optional[int] = None) -> List[List[str]]:
        n_neighbors = n_neighbors or self._num_neighbors
        embeddings = self.get_embeddings(data)
        indexes = self._knn.kneighbors(embeddings, n_neighbors=n_neighbors).indexes
        return self._data[indexes].tolist()

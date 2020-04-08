from enum import Enum
from typing import Optional, Tuple

import nmslib
import numpy as np
from sklearn.preprocessing import normalize as l2_norm


class SpaceName(str, Enum):
    COSINE = "cosinesimil"
    EUCLIDIAN = "l2"


class ApproxKNN:
    def __init__(
        self,
        space: SpaceName = SpaceName.EUCLIDIAN,
        n_neighbors: int = 1,
        ef_construction: int = 1700,
        n_trees: int = 45,
        post: int = 2,
        skip_optimized_index: int = 0,
        ef_search: int = 700,
        normalize: bool = False,
        n_jobs: int = -1,
    ) -> None:
        self.__space = space
        self.__n_neighbors = int(n_neighbors)
        self.__index_params = {
            "M": int(n_trees),
            "efConstruction": int(ef_construction),
            "post": int(post),
            "skip_optimized_index": int(skip_optimized_index),
        }
        self.__ef_search = int(ef_search)
        self.__normalize = normalize
        self.__num_threads = 0 if n_jobs < 0 else n_jobs
        self.__index = nmslib.init(
            method="hnsw", space=self.__space, data_type=nmslib.DataType.DENSE_VECTOR
        )

    def set_query_params(self, ef_search: int) -> None:
        self.__index.setQueryTimeParams({"efSearch": int(ef_search)})

    def fit(self, data: np.ndarray) -> "ApproxKNN":
        if self.__normalize:
            data = l2_norm(data, norm="l2", axis=1)
        self.__index.addDataPointBatch(data)
        self.__index.createIndex(self.__index_params)
        self.set_query_params(self.__ef_search)
        return self

    def kneighbors(
        self, data: np.ndarray, n_neighbors: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.__normalize:
            data = l2_norm(data, norm="l2", axis=1)
        num_neighbors = n_neighbors or self.__n_neighbors
        # TODO: sometimes does not find anything
        neigh_and_dist = np.array(
            self.__index.knnQueryBatch(
                data, k=num_neighbors, num_threads=self.__num_threads
            )
        )
        distances = neigh_and_dist[:, 1, :]
        indexes = neigh_and_dist[:, 0, :].astype(np.int32)
        return distances, indexes

    def load(self, index_path: str) -> "ApproxKNN":
        self.__index.loadIndex(
            index_path, load_data=(self.__index_params["skip_optimized_index"] == 1)
        )
        self.set_query_params(self.__ef_search)
        return self

    def save(self, index_path: str) -> None:
        # we need to save data for non-optimized index
        self.__index.saveIndex(
            index_path, save_data=(self.__index_params["skip_optimized_index"] == 1)
        )

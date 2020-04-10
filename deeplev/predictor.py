from copy import deepcopy
from typing import Dict, List, Callable, Sequence

import numpy as np
import torch
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.fields import ArrayField
from allennlp.predictors.predictor import Predictor
from allennlp.data.dataset import Batch

from deeplev.model import DeepLevenshtein
from deeplev.dataset import LevenshteinReader
from deeplev.utils import clean_sequence
from deeplev.knn import ApproxKNN


@Predictor.register("deep_levenshtein")
class DeepLevenshteinPredictor(Predictor):
    def __init__(
        self,
        model: DeepLevenshtein,
        dataset_reader: LevenshteinReader,
        preprocessor: Callable[[str], str] = clean_sequence,
    ) -> None:
        super().__init__(model=model, dataset_reader=dataset_reader)
        self._preprocessor = preprocessor
        self._knn = ApproxKNN()
        self._data: List[str] = None

    def fit(self, data: Sequence[str]) -> "DeepLevenshteinPredictor":
        data = list(map(self._preprocessor, data))
        instances = [self._dataset_reader.text_to_instance(d, "") for d in data]
        batch = Batch(instances)
        batch.index_instances(self._model.vocab)
        inputs = batch.as_tensor_dict()

        with torch.no_grad():
            vectors = self._model.encode_sequence(inputs["sequence_a"]).cpu().numpy()

        self._data = np.array(data)
        self._knn.fit(vectors)
        return self

    def find_neighbors(self, data: Sequence[str], n_neighbors: int = 10) -> List[str]:
        data = list(map(self._preprocessor, data))
        instances = [self._dataset_reader.text_to_instance(d, "") for d in data]
        batch = Batch(instances)
        batch.index_instances(self._model.vocab)
        inputs = batch.as_tensor_dict()

        with torch.no_grad():
            vectors = self._model.encode_sequence(inputs["sequence_a"]).cpu().numpy()

        _, indexes = self._knn.kneighbors(vectors, n_neighbors=n_neighbors)
        return self._data[indexes].tolist()

    def predict(self, sequence_a: str, sequence_b: str) -> JsonDict:
        return self.predict_json({"sequence_a": sequence_a, "sequence_b": sequence_b})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"label"`` to the output.
        """
        sequence_a = json_dict["sequence_a"]
        sequence_b = json_dict["sequence_b"]
        return self._dataset_reader.text_to_instance(sequence_a, sequence_b)

    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, np.ndarray]
    ) -> List[Instance]:
        new_instance = deepcopy(instance)
        euclidian_distance = outputs["euclidian_distance"]
        new_instance.add_field("distance", ArrayField(array=np.array([euclidian_distance])))
        return [new_instance]

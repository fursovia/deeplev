from typing import Callable

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

from deeplev.model import DeepLevenshtein
from deeplev.dataset import LevenshteinReader
from deeplev.utils import clean_sequence


class DeepLevenshteinPredictor(Predictor):
    def __init__(self, model: DeepLevenshtein, preprocessor: Callable[[str], str] = clean_sequence,) -> None:
        super().__init__(model=model, dataset_reader=LevenshteinReader())
        self._preprocessor = preprocessor

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        raise NotImplementedError

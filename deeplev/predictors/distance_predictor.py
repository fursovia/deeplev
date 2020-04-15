from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

from deeplev.predictors import DeepLevenshteinPredictor


@Predictor.register("edit_distance")
class DistancePredictor(DeepLevenshteinPredictor):
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sequence_a = self._preprocessor(json_dict["sequence_a"])
        sequence_b = self._preprocessor(json_dict["sequence_b"])
        return self._dataset_reader.text_to_instance(sequence_a, sequence_b)

    def edit_distance(self, sequence_a: str, sequence_b: str) -> float:
        return self.predict_json(inputs={"sequence_a": sequence_a, "sequence_b": sequence_b})["euclidian_distance"]

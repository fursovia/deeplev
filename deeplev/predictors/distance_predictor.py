from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor


@Predictor.register("edit_distance")
class DistancePredictor(Predictor):
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        instance = self._dataset_reader.text_to_instance(json_dict["sequence_a"], json_dict["sequence_b"])
        return instance

    def calculate_edit_distance(self, sequence_a: str, sequence_b: str) -> float:
        outputs = self.predict_json(inputs={"sequence_a": sequence_a, "sequence_b": sequence_b})
        edit_distance = outputs["edit_distance"]
        return edit_distance

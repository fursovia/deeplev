from copy import deepcopy
from typing import Dict, List

import numpy as np
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.fields import ArrayField
from allennlp.predictors.predictor import Predictor

from deeplev.dataset import str_to_textfield


@Predictor.register("deep_levenshtein")
class DeepLevenshteinPredictor(Predictor):
    def predict(self, sequence_a: str, sequence_b: str) -> JsonDict:
        return self.predict_json({"sequence_a": sequence_a, "sequence_b": sequence_b})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"label"`` to the output.
        """
        sequence_a = json_dict["sequence_a"]
        sequence_b = json_dict["sequence_b"]

        fields = {}
        fields['anchor'] = str_to_textfield(self._dataset_reader._tokenizer, sequence_a)
        fields['positive'] = str_to_textfield(self._dataset_reader._tokenizer, sequence_b)

        return Instance(fields)

    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, np.ndarray]
    ) -> List[Instance]:
        new_instance = deepcopy(instance)
        new_instance.add_field("similarity", ArrayField(array=np.array([outputs["euclidian_pos"]])))
        return [new_instance]

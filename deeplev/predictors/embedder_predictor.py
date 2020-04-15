from allennlp.common.util import JsonDict, sanitize
from allennlp.predictors.predictor import Predictor
from allennlp.data import Instance
import numpy as np

from deeplev.predictors import DeepLevenshteinPredictor
from deeplev.dataset import sequence_to_textfield


@Predictor.register("embedder")
class EmbedderPredictor(DeepLevenshteinPredictor):
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        fields = {
            "sequence": sequence_to_textfield(
                sequence=self._preprocessor(json_dict["sequence"]), tokenizer=self._dataset_reader.tokenizer
            )
        }
        return Instance(fields=fields)

    def predict_instance(self, instance: Instance) -> JsonDict:
        model_input = self._model.instances_to_model_input([instance])
        outputs = self._model.encode_sequence(**model_input)
        return sanitize(outputs)

    def get_embeddings(self, sequence: str) -> np.ndarray:
        return np.array(self.predict_json(inputs={"sequence": sequence})["vector"]).flatten()

from typing import Sequence

from allennlp.common.util import JsonDict, sanitize
from allennlp.predictors.predictor import Predictor
from allennlp.data import Instance
from allennlp.data.fields import TextField
import numpy as np


@Predictor.register("embedder")
class EmbedderPredictor(Predictor):
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        fields = {
            "sequence": TextField(
                self._dataset_reader.tokenizer.tokenize(json_dict["sequence"]), self._dataset_reader.token_indexers
            )
        }
        return Instance(fields=fields)

    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.encode_on_instances([instance])
        return sanitize(outputs)

    def get_embeddings(self, sequences: Sequence[str]) -> np.ndarray:
        embeddings = []
        for seq in sequences:
            embeddings.extend(self.predict_json(inputs={"sequence": seq}))
        return np.array(embeddings)

from typing import Dict, Union, Iterator

import numpy as np
import pandas as pd
from pathlib import Path
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import ArrayField, Field, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.data.tokenizers.tokenizer import Tokenizer


def _get_default_indexer() -> SingleIdTokenIndexer:
    return SingleIdTokenIndexer(namespace="tokens", start_tokens=[START_SYMBOL], end_tokens=[END_SYMBOL])


def str_to_textfield(tokenizer: Tokenizer, text: str) -> TextField:
    return TextField(
        tokenizer.tokenize(text),
        {
            "tokens": _get_default_indexer()
        }
    )


def float_to_arrayfield(self, value: float) -> ArrayField:
    return ArrayField(
        array=np.array([value])
    )


class LevenshteinReader(DatasetReader):
    def __init__(self, lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = CharacterTokenizer()

    def _read(self, file_path: Union[str, Path]) -> Iterator[Instance]:
        for row in pd.read_csv(file_path, chunksize=1):
            yield self.text_to_instance(
                anchor=row.anchor.squeeze(),
                positive=row.positive.squeeze(),
                negative=row.negative.squeeze(),
                positive_distance=row.positive_distance.squeeze(),
                negative_distance=row.negative_distance.squeeze(),
                inbetween_distance=row.inbetween_distance.squeeze()
            )

    def text_to_instance(
        self,
        anchor: str,
        positive: str,
        negative: str,
        positive_distance: float,
        negative_distance: float,
        inbetween_distance: float
    ) -> Instance:
        fields: Dict[str, Field] = dict()

        fields["anchor"] = str_to_textfield(self._tokenizer, anchor)
        fields["positive"] = str_to_textfield(self._tokenizer, positive)
        fields["negative"] = str_to_textfield(self._tokenizer, negative)
        fields["positive_distance"] = float_to_arrayfield(self._tokenizer, positive_distance)
        fields["negative_distance"] = float_to_arrayfield(self._tokenizer, negative_distance)
        fields["inbetween_distance"] = float_to_arrayfield(self._tokenizer, inbetween_distance)

        return Instance(fields)

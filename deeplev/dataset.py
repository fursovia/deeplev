from typing import Dict, Optional, Union, Iterator
import csv

import numpy as np
import pandas as pd
from pathlib import Path
from allennlp.data import Instance
from allennlp.data.fields import TextField, Field, ArrayField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.common.util import START_SYMBOL, END_SYMBOL


def _get_default_indexer() -> SingleIdTokenIndexer:
    return SingleIdTokenIndexer(namespace='tokens', start_tokens=[START_SYMBOL], end_tokens=[END_SYMBOL])


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

    def str_to_textfield(self, text: str) -> TextField:
        return TextField(
            self._tokenizer.tokenize(text),
            {
                "tokens": _get_default_indexer()
            }
        )

    def float_to_arrayfield(self, value: float) -> ArrayField:
        return ArrayField(
            array=np.array([value])
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

        fields["anchor"] = self.str_to_textfield(anchor)
        fields["positive"] = self.str_to_textfield(positive)
        fields["negative"] = self.str_to_textfield(negative)
        fields["positive_distance"] = self.float_to_arrayfield(positive_distance)
        fields["negative_distance"] = self.float_to_arrayfield(negative_distance)
        fields["inbetween_distance"] = self.float_to_arrayfield(inbetween_distance)

        return Instance(fields)

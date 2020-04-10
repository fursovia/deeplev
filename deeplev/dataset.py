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
from allennlp.common.file_utils import cached_path
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

        fields["anchor"] = TextField(
            self._tokenizer.tokenize(anchor),
            {
                "tokens": _get_default_indexer()
            }
        )
        fields["positive"] = TextField(
            self._tokenizer.tokenize(positive),
            {
                "tokens": _get_default_indexer()
            }
        )
        fields["negative"] = TextField(
            self._tokenizer.tokenize(negative),
            {
                "tokens": _get_default_indexer()
            }
        )
        fields["positive_distance"] = ArrayField(
            array=np.array([positive_distance])
        )
        fields["negative_distance"] = ArrayField(
            array=np.array([negative_distance])
        )
        fields["inbetween_distance"] = ArrayField(
            array=np.array([inbetween_distance])
        )

        return Instance(fields)

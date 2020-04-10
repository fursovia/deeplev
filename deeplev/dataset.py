import csv
from typing import Dict, Optional

import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import ArrayField, Field, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import CharacterTokenizer


def _get_default_indexer() -> SingleIdTokenIndexer:
    return SingleIdTokenIndexer(namespace="tokens", start_tokens=[START_SYMBOL], end_tokens=[END_SYMBOL])


class LevenshteinReader(DatasetReader):
    def __init__(self, lazy: bool = False):
        super().__init__(lazy)
        self._tokenizer = CharacterTokenizer()

    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            tsv_in = csv.reader(data_file, delimiter=",")
            next(tsv_in, None)
            for row in tsv_in:
                if len(row) == 3:
                    yield self.text_to_instance(sequence_a=row[0], sequence_b=row[1], distance=row[2])
                else:
                    yield self.text_to_instance(sequence_a=row[0], sequence_b=row[1])

    def text_to_instance(self, sequence_a: str, sequence_b: str, distance: Optional[float] = None) -> Instance:
        fields: Dict[str, Field] = dict()
        fields["sequence_a"] = TextField(self._tokenizer.tokenize(sequence_a), {"tokens": _get_default_indexer()})

        fields["sequence_b"] = TextField(self._tokenizer.tokenize(sequence_b), {"tokens": _get_default_indexer()})

        if distance is not None:
            # TODO: fix this hack
            fields["distance"] = ArrayField(array=np.array([distance]))

        return Instance(fields)

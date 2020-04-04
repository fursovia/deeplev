from typing import List, Dict, Optional
import csv

import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, Field, ArrayField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL


def _get_default_indexer() -> SingleIdTokenIndexer:
    return SingleIdTokenIndexer(namespace='tokens', start_tokens=[START_SYMBOL], end_tokens=[END_SYMBOL])


class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[Token]:
        return [Token(t) for t in text.split()]


class LevenshteinReader(DatasetReader):
    def __init__(self, lazy: bool = False):
        super().__init__(lazy)
        self._tokenizer = WhitespaceTokenizer()

    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            tsv_in = csv.reader(data_file, delimiter=',')
            next(tsv_in, None)
            for row in tsv_in:
                if len(row) == 3:
                    yield self.text_to_instance(sequence_a=row[0], sequence_b=row[1], similarity=row[2])
                else:
                    yield self.text_to_instance(sequence_a=row[0], sequence_b=row[1])

    def text_to_instance(
        self,
        sequence_a: str,
        sequence_b: str,
        similarity: Optional[float] = None
    ) -> Instance:
        fields: Dict[str, Field] = dict()
        fields["sequence_a"] = TextField(
            self._tokenizer.tokenize(sequence_a),
            {
                "tokens": _get_default_indexer()
            }
        )

        fields["sequence_b"] = TextField(
            self._tokenizer.tokenize(sequence_b),
            {
                "tokens": _get_default_indexer()
            }
        )

        if similarity is not None:
            # TODO: fix this hack
            fields["distance"] = ArrayField(
                array=np.array([similarity])
            )

        return Instance(fields)

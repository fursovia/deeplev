from typing import Optional, Dict
import jsonlines

import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data.fields import TextField, ArrayField
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.data import DatasetReader, Instance, Field, Tokenizer, TokenIndexer


@DatasetReader.register("levenshtein")
class LevenshteinReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        tokenizer: Tokenizer = CharacterTokenizer(),
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self.token_indexers = token_indexers
        self.tokenizer = tokenizer

    def _read(self, file_path):
        with jsonlines.open(cached_path(file_path), "r") as reader:
            for items in reader:
                instance = self.text_to_instance(
                    sequence_a=items["sequence_a"], sequence_b=items["sequence_b"], distance=items.get("distance")
                )
                yield instance

    def text_to_instance(self, sequence_a: str, sequence_b: str, distance: Optional[float] = None) -> Instance:
        fields: Dict[str, Field] = dict()
        fields["sequence_a"] = TextField(self.tokenizer.tokenize(sequence_a), self.token_indexers)
        fields["sequence_b"] = TextField(self.tokenizer.tokenize(sequence_b), self.token_indexers)

        if distance is not None:
            fields["distance"] = ArrayField(array=np.array([distance]))

        return Instance(fields)

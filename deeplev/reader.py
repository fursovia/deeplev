from typing import Optional, Dict
import logging
import jsonlines
import math

import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data.fields import TextField, ArrayField
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.data import DatasetReader, Instance, Field, Tokenizer, TokenIndexer

logger = logging.getLogger(__name__)


@DatasetReader.register("levenshtein")
class LevenshteinReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        tokenizer: Tokenizer = CharacterTokenizer(),
        max_sequence_length: Optional[int] = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self.token_indexers = token_indexers
        self.tokenizer = tokenizer
        self._max_sequence_length = max_sequence_length or math.inf

        logger.info("Creating LevenshteinReader")
        logger.info("max_sequence_length=%s", max_sequence_length)

    def _read(self, file_path):

        logger.info("Loading data from %s", file_path)
        dropped_instances = 0

        with jsonlines.open(cached_path(file_path), "r") as reader:
            for items in reader:
                instance = self.text_to_instance(
                    sequence_a=items["sequence_a"], sequence_b=items["sequence_b"], distance=items.get("distance")
                )

                if (
                    instance.fields["sequence_a"].sequence_length() <= self._max_sequence_length
                    and instance.fields["sequence_b"].sequence_length() <= self._max_sequence_length
                ):
                    yield instance
                else:
                    dropped_instances += 1

        if not dropped_instances:
            logger.info(f"No instances dropped from {file_path}.")
        else:
            logger.warning(f"Dropped {dropped_instances} instances from {file_path}.")

    def text_to_instance(self, sequence_a: str, sequence_b: str, distance: Optional[float] = None) -> Instance:
        fields: Dict[str, Field] = {}
        tokens_a = self.tokenizer.tokenize(sequence_a)
        tokens_b = self.tokenizer.tokenize(sequence_b)

        fields["sequence_a"] = TextField(tokens_a, self.token_indexers)
        fields["sequence_b"] = TextField(tokens_b, self.token_indexers)

        if distance is not None:
            fields["distance"] = ArrayField(array=np.array([distance]))

        return Instance(fields)

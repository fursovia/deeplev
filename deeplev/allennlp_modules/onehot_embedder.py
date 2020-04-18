from typing import Optional

import torch
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.common import Params
from allennlp.data import Vocabulary


@TokenEmbedder.register("onehot_embedder")
class OnehotEmbedder(TokenEmbedder):
    def __init__(self, vocab_size: int, max_seq_length: Optional[int] = None) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._max_seq_length = max_seq_length

    def get_output_dim(self) -> int:
        return self._vocab_size

    def _one_hot(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(tensor, self._vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        num_tokens = tokens.size(1)
        one_hot_labels = self._one_hot(tokens)
        if self._max_seq_length is not None:
            pad_length = self._max_seq_length - num_tokens
            one_hot_labels = torch.nn.functional.pad(one_hot_labels, pad=[0, 0, 0, pad_length])
        return one_hot_labels

    @classmethod
    def from_params(cls, params: Params, vocab: Vocabulary) -> "OnehotEmbedder":
        vocab_size = params.pop_int("vocab_size", None)
        vocab_namespace = params.pop("vocab_namespace", None if vocab_size else "tokens")
        if vocab_size is None:
            vocab_size = vocab.get_vocab_size(vocab_namespace)

        max_seq_length = params.pop_int("max_seq_length", None)
        params.assert_empty(cls.__name__)

        return cls(vocab_size=vocab_size, max_seq_length=max_seq_length)

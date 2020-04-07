from typing import Optional

import torch
from allennlp.modules.token_embedders import TokenEmbedder


@TokenEmbedder.register("onehot_encoder")
class OnehotEncoder(TokenEmbedder):
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
            one_hot_labels = torch.nn.functional.pad(
                one_hot_labels, pad=[0, 0, 0, pad_length]
            )
        return one_hot_labels

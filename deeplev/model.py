from typing import Dict, Optional, List

import torch
from allennlp.data import Vocabulary, Instance, Batch, TextFieldTensors
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models import Model
from allennlp.nn import util


@Model.register("deep_levenshtein")
class DeepLevenshtein(Model):
    """
    This `Model` implements a framework for fast approximate similarity search
    as described in [Convolutional Embedding for Edit Distance](https://arxiv.org/abs/2001.11692).
    The `DeepLevenshtein` takes two sequences (texts) `x` and `y` as input as outputs an
    approximated edit-distance score between `x` and `y`.

    Firstly, the model starts by embedding the tokens via `text_field_embedder`.
    Then, we encode these embeddings with a `Seq2SeqEncoder` (`seq2seq_encoder`) (if given).
    We obtain vector representations of the sequences `z_x` and `z_y`
    using `Seq2VecEncoder` (`seq2vec_encoder`). Lastly, we calculate pairwise euclidian distance
    between all vector representations of the sequences in the batch.

    Registered as a `Model` with name "deep_levenshtein".

    # Parameters
    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : `TextFieldEmbedder`, required
        Used to embed the `tokens` `TextField` we get as input to the model.
    seq2vec_encoder : `Seq2VecEncoder`
        The sequence-to-vector encoder to use on the tokens.
    seq2seq_encoder : `Seq2SeqEncoder`
        The sequence-to-sequence encoder to use for additional
        bunch of layers
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
    ) -> None:
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.seq2seq_encoder = seq2seq_encoder
        self.seq2vec_encoder = seq2vec_encoder

        self._loss = torch.nn.MSELoss()

    def encode_sequence(self, sequence: TextFieldTensors) -> torch.Tensor:
        embedded_sequence = self.text_field_embedder(sequence)
        mask = util.get_text_field_mask(sequence).float()
        # It is needed if we pad the initial sequence (or truncate)
        mask = torch.nn.functional.pad(mask, pad=[0, embedded_sequence.size(1) - mask.size(1)])
        if self.seq2seq_encoder is not None:
            embedded_sequence = self.seq2seq_encoder(embedded_sequence, mask=mask)
        embedded_sequence_vector = self.seq2vec_encoder(embedded_sequence, mask=mask)
        return embedded_sequence_vector

    def encode_on_instances(self, instances: List[Instance]) -> torch.Tensor:
        inputs = self._instances_to_model_input(instances)
        embedded_sequence_vector = self.encode_sequence(**inputs)
        return embedded_sequence_vector

    def _instances_to_model_input(self, instances: List[Instance]) -> Dict[str, torch.Tensor]:
        cuda_device = self._get_prediction_device()
        dataset = Batch(instances)
        dataset.index_instances(self.vocab)
        model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
        return model_input

    def forward(
        self, sequence_a: TextFieldTensors, sequence_b: TextFieldTensors, distance: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        embedded_sequence_a = self.encode_sequence(sequence_a)
        embedded_sequence_b = self.encode_sequence(sequence_b)

        euclidian_distance = torch.pairwise_distance(embedded_sequence_a, embedded_sequence_b)
        output_dict = {
            "edit_distance": euclidian_distance,
            "emb_sequence_a": embedded_sequence_a,
            "emb_sequence_b": embedded_sequence_b,
        }

        if distance is not None:
            loss = self._loss(euclidian_distance, distance.view(-1))
            output_dict["loss"] = loss

        return output_dict

from typing import Dict, Optional, List

import torch
from allennlp.data.dataset import Batch
from allennlp.data import Vocabulary, Instance
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn import util


class DeepLevenshtein(Model):
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

        self._loss = torch.nn.L1Loss()

    def encode_sequence(self, sequence: Dict[str, torch.LongTensor]) -> torch.Tensor:
        embedded_sequence = self.text_field_embedder(sequence)
        mask = util.get_text_field_mask(sequence).float()
        # It is needed if we pad the initial sequence (or truncate)
        mask = torch.nn.functional.pad(mask, pad=[0, embedded_sequence.size(1) - mask.size(1)])
        if self.seq2seq_encoder is not None:
            embedded_sequence = self.seq2seq_encoder(embedded_sequence, mask=mask)
        embedded_sequence_vector = self.seq2vec_encoder(embedded_sequence, mask=mask)
        return embedded_sequence_vector

    def forward(
        self,
        sequence_a: Dict[str, torch.LongTensor],
        sequence_b: Dict[str, torch.LongTensor],
        distance: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        embedded_sequence_a = self.encode_sequence(sequence_a)
        embedded_sequence_b = self.encode_sequence(sequence_b)

        euclidian_distance = torch.pairwise_distance(embedded_sequence_a, embedded_sequence_b)
        output_dict = {"euclidian_distance": euclidian_distance}

        if distance is not None:
            loss = self._loss(euclidian_distance, distance.view(-1))
            output_dict["loss"] = loss

        return output_dict

    def instances_to_model_input(self, instances: List[Instance]) -> Dict[str, torch.Tensor]:
        cuda_device = self._get_prediction_device()
        dataset = Batch(instances)
        dataset.index_instances(self.vocab)
        model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
        return model_input

from typing import Dict, Optional

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.attention import Attention
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
        attention: Optional[Attention] = None,
    ) -> None:
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.seq2seq_encoder = seq2seq_encoder
        self.seq2vec_encoder = seq2vec_encoder
        self.attention = attention

        self._loss = torch.nn.L1Loss()

    def prepare_attended_input(
        self, seq_attention_from: Dict[str, torch.Tensor], seq_attention_to: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        input_weights = self.attention(
            seq_attention_from["vector"], seq_attention_to["matrix"], seq_attention_to["mask"],
        )
        attended_input = util.weighted_sum(seq_attention_to["matrix"], input_weights)
        attented_seq = torch.cat((seq_attention_from["vector"], attended_input), -1)
        return attented_seq

    def calculate_euclidian_distance(
        self, embedded_sequence_a: Dict[str, torch.Tensor], embedded_sequence_b: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.attention:
            vector_a = self.prepare_attended_input(embedded_sequence_a, embedded_sequence_b)
            vector_b = self.prepare_attended_input(embedded_sequence_b, embedded_sequence_a)
        else:
            vector_a = embedded_sequence_a["vector"]
            vector_b = embedded_sequence_b["vector"]
        return torch.pairwise_distance(vector_a, vector_b, p=2.0)

    def encode_sequence(self, sequence: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        embedded_sequence = self.text_field_embedder(sequence)
        mask = util.get_text_field_mask(sequence).float()
        # It is needed if we pad the initial sequence
        mask = torch.nn.functional.pad(mask, pad=[0, embedded_sequence.size(1) - mask.size(1)])
        if self.seq2seq_encoder is not None:
            embedded_sequence = self.seq2seq_encoder(embedded_sequence, mask=mask)
        embedded_sequence_vector = self.seq2vec_encoder(embedded_sequence, mask=mask)
        return {
            "mask": mask,
            "vector": embedded_sequence_vector,
            "matrix": embedded_sequence,
        }

    def forward(
        self,
        sequence_a: Dict[str, torch.LongTensor],
        sequence_b: Dict[str, torch.LongTensor],
        distance: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        embedded_sequence_a = self.encode_sequence(sequence_a)
        embedded_sequence_b = self.encode_sequence(sequence_b)

        euclidian_distance = self.calculate_euclidian_distance(embedded_sequence_a, embedded_sequence_b)
        output_dict = {"euclidian_distance": euclidian_distance}

        if distance is not None:
            loss = self._loss(euclidian_distance, distance.view(-1))
            output_dict["loss"] = loss

        return output_dict

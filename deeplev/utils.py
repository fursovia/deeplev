import functools
import re

import torch
from allennlp.models import Model
import Levenshtein as lvs


def load_weights(model: Model, path: str, location: str = 'cpu') -> None:
    with open(path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=location))


@functools.lru_cache(maxsize=500)
def edit_distance(sequence_a: str, sequence_b: str) -> float:
    return lvs.distance(sequence_a, sequence_b)


def clean_sequence(sequence: str) -> str:
    sequence = sequence.lower()
    sequence = re.sub(r"[^\w0-9 ]+", "", sequence)
    sequence = re.sub("\s\s+", " ", sequence).strip()
    return sequence

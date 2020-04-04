import functools

import torch
from allennlp.models import Model
import Levenshtein as lvs


def load_weights(model: Model, path: str, location: str = 'cpu') -> None:
    with open(path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=location))


@functools.lru_cache(maxsize=500)
def edit_distance(sequence_a: str, sequence_b: str) -> float:
    return lvs.distance(sequence_a, sequence_b)

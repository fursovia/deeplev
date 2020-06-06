import functools
from multiprocessing import Pool
from typing import Sequence
import re

import Levenshtein as lvs
import torch
import numpy as np
from tqdm import tqdm
from allennlp.models import Model


def load_weights(model: Model, path: str, location: str = "cpu") -> None:
    with open(path, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=location))


def clean_sequence(sequence: str) -> str:
    sequence = sequence.lower()
    sequence = re.sub(r"[^\w0-9 ]+", "", sequence)
    sequence = re.sub(r"\s\s+", " ", sequence).strip()
    return sequence


@functools.lru_cache(maxsize=500)
def edit_distance(sequence_a: str, sequence_b: str) -> float:
    return lvs.distance(sequence_a, sequence_b)


def _edit_distance_one_vs_all(x):
    a, bs = x
    return [edit_distance(a, b) for b in bs]


def pairwise_edit_distances(
    sequences_a: Sequence[str], sequences_b: Sequence[str], n_jobs: int = 5, verbose: bool = False
) -> np.ndarray:
    bar = tqdm if verbose else lambda iterable, total, desc: iterable

    with Pool(n_jobs) as pool:
        distances = list(
            bar(
                pool.imap(_edit_distance_one_vs_all, zip(sequences_a, [sequences_b for _ in sequences_a])),
                total=len(sequences_a),
                desc="# edit distance {}x{}".format(len(sequences_a), len(sequences_b)),
            )
        )
    return np.array(distances)

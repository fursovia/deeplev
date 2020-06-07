import functools
from multiprocessing import Pool
from typing import Sequence, Dict, Any, List, Callable, Union, Optional
import jsonlines
import re

import Levenshtein as lvs
import torch
import numpy as np
from tqdm import tqdm
from allennlp.models import Model

from deeplev.modules.typo_generator import generate_default_typo


def load_weights(model: Model, path: str, location: str = "cpu") -> None:
    with open(path, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=location))


def clean_sequence(sequence: str) -> str:
    sequence = sequence.lower()
    sequence = re.sub(r"[^\w0-9 ]+", "", sequence)
    sequence = re.sub(r"\s\s+", " ", sequence).strip()
    return sequence


@functools.lru_cache(maxsize=500)
def edit_distance(sequence_a: str, sequence_b: str) -> int:
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


def load_jsonlines(path: str) -> List[Dict[str, Any]]:
    data = []
    with jsonlines.open(path, "r") as reader:
        for items in reader:
            data.append(items)
    return data


def write_jsonlines(data: Sequence[Dict[str, Any]], path: str) -> None:
    with jsonlines.open(path, "w") as writer:
        for ex in data:
            writer.write(ex)


def create_dataset(
    sequences: Sequence[str],
    cleaner: Callable[[str], str] = clean_sequence,
    num_original: Optional[int] = None,
    num_artificial: Optional[int] = None,
) -> List[Dict[str, Union[str, int]]]:
    sequences = list(map(cleaner, sequences))
    vocab = list(set("".join(sequences)))
    num_original = num_original or len(sequences)
    num_artificial = num_artificial or len(sequences)

    dissimilar_examples = []
    dissimilar_indexes = np.random.randint(0, len(sequences), size=(num_original, 2))
    for id1, id2 in tqdm(dissimilar_indexes):
        seq_a = sequences[id1]
        seq_b = sequences[id2]
        dist = edit_distance(seq_a, seq_b)
        dissimilar_examples.append(dict(sequence_a=seq_a, sequence_b=seq_b, distance=dist))

    similar_examples = []
    similar_indexes = np.random.randint(0, len(sequences), size=num_artificial)
    for idx in tqdm(similar_indexes):
        seq_a = sequences[idx]
        seq_b = generate_default_typo(seq_a, vocab)
        if seq_a != seq_b:
            dist = edit_distance(seq_a, seq_b)
            similar_examples.append(dict(sequence_a=seq_a, sequence_b=seq_b, distance=dist))

    examples = dissimilar_examples + similar_examples
    return examples

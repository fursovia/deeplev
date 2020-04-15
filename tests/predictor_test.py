from typing import List, Dict

from allennlp.data.vocabulary import Vocabulary
from pytest import approx

from deeplev.predictors import DistancePredictor, EmbedderPredictor, SearchPredictor
from deeplev.model import get_deep_levenshtein


def _get_dummy_counter(sequences: List[str]) -> Dict[str, Dict[str, int]]:
    counter = {}
    vocab = set("".join(sequences))
    for s in vocab:
        counter[s] = 1
    return {"tokens": counter}


def test_zero_distance():
    sequence = "abcdefg"
    model = get_deep_levenshtein(vocab=Vocabulary(counter=_get_dummy_counter([sequence])))
    predictor = DistancePredictor(model=model)
    approx_dist = predictor.edit_distance(sequence, sequence)
    assert approx_dist == approx(0.0, abs=1e-5)


def test_non_zero_distance():
    sequence_a = "abcdefg"
    sequence_b = "qwerty"
    model = get_deep_levenshtein(vocab=Vocabulary(counter=_get_dummy_counter([sequence_a, sequence_b])))
    predictor = DistancePredictor(model=model)
    approx_dist = predictor.edit_distance(sequence_a, sequence_b)
    assert approx_dist > 0.0


def test_embedder_predictor():
    sequence = "qwerty"
    model = get_deep_levenshtein(vocab=Vocabulary(counter=_get_dummy_counter([sequence])))
    predictor = EmbedderPredictor(model=model)
    embedding = predictor.get_embeddings(sequence=sequence)
    assert len(embedding.shape) == 1


def test_search_predictor():
    sequences = ["asd", "sdf", "dfg"]

    model = get_deep_levenshtein(vocab=Vocabulary(counter=_get_dummy_counter(sequences)))
    predictor = SearchPredictor(model=model, num_neighbors=1)
    predictor.fit(sequences)
    neighbors = predictor.find_neighbors(sequences)
    for i, pred in enumerate(neighbors):
        assert pred[0] == sequences[i]

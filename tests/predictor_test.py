from typing import List, Dict
from pathlib import Path

from allennlp.data.vocabulary import Vocabulary
from allennlp.common import Params
from pytest import approx

from deeplev.predictors import DistancePredictor, EmbedderPredictor, SearchPredictor
from deeplev.model import DeepLevenshtein


PROJECT_ROOT = (Path(__file__).parent / "..").resolve()
_DEFAULT_CONFIG_PATH = PROJECT_ROOT / "model_config/bilstm.jsonnet"


def _get_dummy_counter(sequences: List[str]) -> Dict[str, Dict[str, int]]:
    counter = {}
    vocab = set("".join(sequences))
    for s in vocab:
        counter[s] = 1
    return {"tokens": counter}


def test_zero_distance():
    sequence = "abcdefg"
    model = DeepLevenshtein.from_params(
        params=Params.from_file(_DEFAULT_CONFIG_PATH), vocab=Vocabulary(counter=_get_dummy_counter([sequence]))
    )
    predictor = DistancePredictor(model=model)
    approx_dist = predictor.calculate_edit_distance(sequence, sequence)
    assert approx_dist == approx(0.0, abs=1e-5)


def test_non_zero_distance():
    sequence_a = "abcdefg"
    sequence_b = "qwerty"
    model = DeepLevenshtein.from_params(
        params=Params.from_file(_DEFAULT_CONFIG_PATH),
        vocab=Vocabulary(counter=_get_dummy_counter([sequence_a, sequence_b])),
    )
    predictor = DistancePredictor(model=model)
    approx_dist = predictor.calculate_edit_distance(sequence_a, sequence_b)
    assert approx_dist > 0.0


def test_embedder_predictor_output_shape():
    sequence = "qwerty"
    model = DeepLevenshtein.from_params(
        params=Params.from_file(_DEFAULT_CONFIG_PATH), vocab=Vocabulary(counter=_get_dummy_counter([sequence]))
    )
    predictor = EmbedderPredictor(model=model)
    embedding = predictor.get_embeddings(sequences=[sequence])
    embeddings = predictor.get_embeddings(sequences=[sequence, sequence])
    assert embedding.shape == (1, 64)
    assert embeddings.shape == (2, 64)


def test_search_predictor_right_top1():
    sequences = ["asd", "sdf", "dfg"]

    model = DeepLevenshtein.from_params(
        params=Params.from_file(_DEFAULT_CONFIG_PATH), vocab=Vocabulary(counter=_get_dummy_counter(sequences))
    )
    predictor = SearchPredictor(model=model, num_neighbors=1)
    predictor.fit(sequences)
    neighbors = predictor.find_neighbors(sequences)
    for i, pred in enumerate(neighbors):
        assert pred[0] == sequences[i]

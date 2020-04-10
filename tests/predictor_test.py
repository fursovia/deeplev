from allennlp.data.vocabulary import Vocabulary
from pytest import approx

from deeplev.predictor import DeepLevenshteinPredictor
from deeplev.model import get_deep_levenshtein
from deeplev.dataset import LevenshteinReader


def test_zero_distance():
    seq_a = "abcdefg"
    seq_b = "abcdefg"

    vocab = Vocabulary()
    model = get_deep_levenshtein(vocab)
    reader = LevenshteinReader()
    predictor = DeepLevenshteinPredictor(model=model, dataset_reader=reader)
    approx_dist = predictor.predict(seq_a, seq_b)["euclidian_distance"]
    assert approx_dist == approx(0.0, abs=1e-5)

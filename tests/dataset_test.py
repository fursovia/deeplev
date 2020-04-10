from deeplev.dataset import LevenshteinReader


def test_dummy_dataset():
    reader = LevenshteinReader()
    assert reader is not None

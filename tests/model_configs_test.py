from pathlib import Path

from allennlp.data import Vocabulary
from allennlp.common import Params

from deeplev.model import DeepLevenshtein


PROJECT_ROOT = (Path(__file__).parent / "..").resolve()


def test_base_configs():

    for config_path in Path(PROJECT_ROOT / "model_config").iterdir():
        try:
            params = Params.from_file(config_path)
            blank_vocab = Vocabulary()
            DeepLevenshtein.from_params(params=params, vocab=blank_vocab)
        except Exception as e:
            raise AssertionError(f"unable to load params from {config_path}, because {e}")

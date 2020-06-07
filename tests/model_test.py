from allennlp.data import Vocabulary, DatasetReader, Batch
from allennlp.common import Params
from allennlp.models import Model, load_archive

import deeplev
from tests import PROJECT_ROOT, FIXTURES_ROOT


class TestDeepLevenshtein:

    model_path = FIXTURES_ROOT / "model.tar.gz"

    def test_from_archive(self):
        try:
            deeplev.DeepLevenshtein.from_archive(self.model_path)
        except Exception as e:
            raise AssertionError(f"unable to load model from {self.model_path}, because {e}")

    def test_from_params(self):

        for config_path in (PROJECT_ROOT / "training_config").iterdir():
            try:
                params = Params.from_file(
                    str(config_path), ext_vars={"TRAIN_DATA_PATH": "", "VALID_DATA_PATH": "", "GPU_ID": "-1"}
                )
                blank_vocab = Vocabulary()
                Model.from_params(params=params["model"], vocab=blank_vocab)
            except Exception as e:
                raise AssertionError(f"unable to load params from {config_path}, because {e}")

    def test_forward_pass_runs_correctly(self):
        archive = load_archive(self.model_path)
        model = archive.model
        reader = DatasetReader.from_params(archive.config["dataset_reader"])
        instances = reader.read(FIXTURES_ROOT / "data" / "train.jsonl")

        batch = Batch(instances)
        batch.index_instances(model.vocab)
        output_dict = model(**batch.as_tensor_dict())

        assert set(output_dict.keys()) == {
            "edit_distance",
            "emb_sequence_a",
            "emb_sequence_b",
            "loss",
        }
        assert output_dict["edit_distance"].shape[0] == 3

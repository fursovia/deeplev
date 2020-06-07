import pytest

from allennlp.common.util import ensure_list
from allennlp.common.params import Params
from allennlp.data import TokenIndexer

from tests import FIXTURES_ROOT
from deeplev.reader import LevenshteinReader


class TestLevenshteinReader:
    data_path = FIXTURES_ROOT / "data" / "train.jsonl"

    def _check_outputs(self, reader):
        instances = reader.read(self.data_path)
        instances = ensure_list(instances)

        instance1 = {"sequence_a": ["a", "b", "c"], "sequence_b": ["a", "b", "c"], "distance": 0}
        instance2 = {"sequence_a": ["a", "b", "c"], "sequence_b": ["a", "b", "d"], "distance": 1}
        instance3 = {"sequence_a": ["q", "w", "e"], "sequence_b": ["q", "a", "z"], "distance": 2}

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["sequence_a"].tokens] == instance1["sequence_a"]
        assert [t.text for t in fields["sequence_b"].tokens] == instance1["sequence_b"]
        assert fields["distance"].array[0] == instance1["distance"]

        fields = instances[1].fields
        assert [t.text for t in fields["sequence_a"].tokens] == instance2["sequence_a"]
        assert [t.text for t in fields["sequence_b"].tokens] == instance2["sequence_b"]
        assert fields["distance"].array[0] == instance2["distance"]

        fields = instances[2].fields
        assert [t.text for t in fields["sequence_a"].tokens] == instance3["sequence_a"]
        assert [t.text for t in fields["sequence_b"].tokens] == instance3["sequence_b"]
        assert fields["distance"].array[0] == instance3["distance"]

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = LevenshteinReader(token_indexers={"tokens": TokenIndexer.by_name("single_id")()}, lazy=False)
        self._check_outputs(reader)

    def test_from_params(self):
        params = Params({"token_indexers": {"tokens": "single_id"}, "tokenizer": "character"})
        reader = LevenshteinReader.from_params(params)
        self._check_outputs(reader)

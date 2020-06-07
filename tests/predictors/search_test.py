import deeplev
from tests import FIXTURES_ROOT


class TestSearchPredictor:

    model_path = FIXTURES_ROOT / "model.tar.gz"

    def test_predictions(self):
        predictor = deeplev.SearchPredictor.from_path(self.model_path, predictor_name="searcher")
        inputs = ["abc", "abd", "abc"]
        predictor.fit(inputs)
        output = predictor.find_neighbors(inputs)

        assert len(output) == 3
        assert output[0][0] == "abc"
        assert output[1][0] == "abd"
        assert output[2][0] == "abc"

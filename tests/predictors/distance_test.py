import pytest

import deeplev
from tests import FIXTURES_ROOT


class TestDistancePredictor:

    model_path = FIXTURES_ROOT / "model.tar.gz"

    def test_predictions(self):
        predictor = deeplev.DistancePredictor.from_path(self.model_path, predictor_name="edit_distance")
        inputs = {"sequence_a": "abc", "sequence_b": "abd"}
        output = predictor.calculate_edit_distance(**inputs)

        assert isinstance(output, float)
        assert output > 0.0

        inputs = {"sequence_a": "abc", "sequence_b": "abc"}
        output = predictor.calculate_edit_distance(**inputs)
        assert output == pytest.approx(0.0, abs=1e-5)

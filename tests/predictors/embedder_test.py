import numpy as np
import deeplev
from tests import FIXTURES_ROOT


class TestEmbedderPredictor:

    model_path = FIXTURES_ROOT / "model.tar.gz"

    def test_predictions(self):
        predictor = deeplev.EmbedderPredictor.from_path(self.model_path, predictor_name="embedder")
        inputs = ["abc", "abd", "abc"]
        output = predictor.get_embeddings(inputs)

        assert len(output.shape) == 2
        assert output.shape[0] == 3
        assert np.linalg.norm(output[0] - output[2]) < 1e-6
        assert np.linalg.norm(output[1] - output[2]) > 1e-4

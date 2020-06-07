import argparse
from pathlib import Path
from collections import defaultdict
import json

import pandas as pd
from allennlp.data import Vocabulary
from allennlp.common import Params

from deeplev.utils import pairwise_edit_distances
from deeplev.predictors import SearchPredictor
from deeplev.model import DeepLevenshtein
from deeplev.utils import load_weights


parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, required=True)
parser.add_argument("--serialization-dir", type=str, required=True)
parser.add_argument("--col-name", type=str, required=True)
parser.add_argument("--samples", type=int, default=None)
parser.add_argument("--n-jobs", type=int, default=5)


TOP_K_VALUES = [1, 3, 5, 10, 20, 30, 40, 50]

if __name__ == "__main__":
    args = parser.parse_args()
    data = pd.read_csv(args.data_path)
    num_samples = args.samples or data.shape[0]
    num_samples = num_samples if num_samples <= data.shape[0] else data.shape[0]
    samples = data.sample(n=num_samples, replace=False)

    database = data[args.col_name].values
    queries = samples[args.col_name].values

    distances = pairwise_edit_distances(
        sequences_a=queries, sequences_b=database, n_jobs=args.n_jobs, verbose=True
    )
    sorted_indexes = distances.argsort(axis=1)
    y_true = database[sorted_indexes].tolist()

    # TODO: replace these 6+ lines with one `from_path`
    # predictor = SearchPredictor.from_path(archive_path=args.serialization_dir)
    serialization_dir = Path(args.serialization_dir)
    vocab = Vocabulary.from_files(serialization_dir / "vocab")
    config = json.load(open(serialization_dir / "args.json"))["config"]
    model = DeepLevenshtein.from_params(params=Params.from_file(config), vocab=vocab)
    load_weights(model, serialization_dir / "best.th")

    predictor = SearchPredictor(model=model, num_neighbors=max(TOP_K_VALUES))
    predictor.fit(data=database)
    y_pred = predictor.find_neighbors(data=queries)

    recall_at_k = defaultdict(float)
    for k in TOP_K_VALUES:
        for true_vals, pred_vals in zip(y_true, y_pred):
            true_vals = true_vals[:k]
            pred_vals = pred_vals[:k]
            recall_at_k[k] += (sum(int(p in true_vals) for p in pred_vals) / k) / args.samples

    for k, recall in recall_at_k.items():
        print(f"Recall@{k} = {recall:.3f}")

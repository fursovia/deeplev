import argparse
from collections import defaultdict

import pandas as pd
from deeplev.utils import pairwise_edit_distances
from deeplev.predictors import SearchPredictor

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--serialization_dir", type=str, required=True)
parser.add_argument("--col_name", type=str, required=True)
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--n_jobs", type=int, default=5)


TOP_K_VALUES = [1, 3, 5, 10, 20, 30, 40, 50]

if __name__ == "__main__":
    args = parser.parse_args()
    data = pd.read_csv(args.data_path)
    num_samples = args.samples if args.samples < data.shape[0] else data.shape[0]
    samples = data.sample(n=num_samples, replace=False)

    database = data[args.col_name].values
    queries = samples[args.col_name].values

    distances = pairwise_edit_distances(
        sequences_a=queries, sequences_b=database, n_jobs=args.n_jobs, verbose=True
    )
    sorted_indexes = distances.argsort(axis=1)
    y_true = database[sorted_indexes].tolist()

    predictor = SearchPredictor.from_path(archive_path=args.serialization_dir)
    predictor.fit(data=database)
    y_pred = predictor.find_neighbors(data=queries, n_neighbors=max(TOP_K_VALUES))

    recall_at_k = defaultdict(float)
    for k in TOP_K_VALUES:
        for t, p in zip(y_true, y_pred):
            t = t[:k]
            p = p[:k]
            recall_at_k[k] += (sum(int(tt in p) for tt in t) / k) / args.samples

    print(recall_at_k)

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from deeplev.utils import create_dataset, write_jsonlines

parser = argparse.ArgumentParser()
parser.add_argument("--csv-path", type=str, required=True)
parser.add_argument("--col-name", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)

parser.add_argument("--num-artificial", type=int, default=50_000)
parser.add_argument("--num-original", type=int, default=50_000)
parser.add_argument("--test_size", type=float, default=0.05)


if __name__ == "__main__":
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    assert not train_path.exists() and not valid_path.exists()

    data = pd.read_csv(args.csv_path)
    data = data[(~data[args.col_name].isna()) & (data[args.col_name].apply(len) > 0)]
    sequences = data[args.col_name].astype(str).tolist()

    examples = create_dataset(sequences, num_original=args.num_original, num_artificial=args.num_artificial)
    train, valid = train_test_split(examples, test_size=args.test_size)

    write_jsonlines(train, train_path)
    write_jsonlines(valid, valid_path)

import argparse
import random
from tqdm import tqdm
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from deeplev.utils import edit_distance
from deeplev.sampler import TypoSampler


parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, required=True)
parser.add_argument('--col_name', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--dataset_size', type=int, default=1_000_000)
parser.add_argument('--test_size', type=float, default=0.05)


if __name__ == "__main__":
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    assert not train_path.exists() and not test_path.exists()

    data = pd.read_csv(args.csv_path)
    data = data[~data[args.col_name].isna()]
    # TODO: drop zero-len examples
    sequences = data[args.col_name].astype(str).tolist()
    sampler = TypoSampler(sequences)

    examples = []

    for i in tqdm(range(args.dataset_size)):
        random_text = random.sample(sequences, 1)[0]
        positive = sampler.sample_positive(random_text)
        negative = sampler.sample_negative(random_text)

        if positive == random_text or negative == random_text:
            continue

        positive_distance = edit_distance(random_text, positive)
        negative_distance = edit_distance(random_text, negative)
        inbetween_distance = edit_distance(positive, negative)

        if positive_distance < negative_distance:
            examples.append((random_text, positive, negative, positive_distance, negative_distance, inbetween_distance))

    examples = pd.DataFrame(
        examples,
        columns=['anchor', 'positive', 'negative', 'positive_distance', 'negative_distance', 'inbetween_distance']
    )

    train, test = train_test_split(examples, test_size=args.test_size)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

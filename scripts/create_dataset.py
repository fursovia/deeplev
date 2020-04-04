import argparse
from tqdm import tqdm
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from deeplev.utils import edit_distance, clean_sequence
from deeplev.typo_generator import generate_default_typo


parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--num_dissimilar', type=int, default=300000)
parser.add_argument('--num_similar', type=int, default=700000)
parser.add_argument('--test_size', type=float, default=0.05)


if __name__ == '__main__':
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    train_path = output_dir / 'train.csv'
    test_path = output_dir / 'test.csv'
    assert not train_path.exists() and not test_path.exists()

    data = pd.read_csv(args.csv_path)
    sequences = data['sequences'].tolist()
    sequences = list(map(clean_sequence, sequences))
    vocab = list(set(''.join(sequences)))

    dissimilar_examples = []
    dissimilar_indexes = np.random.randint(0, len(sequences), size=(args.num_dissimilar, 2))
    for id1, id2 in tqdm(dissimilar_indexes):
        tr1 = sequences[id1]
        tr2 = sequences[id2]
        dist = edit_distance(tr1, tr2)
        dissimilar_examples.append((tr1, tr2, dist))

    similar_examples = []
    similar_indexes = np.random.randint(0, len(sequences), size=args.num_similar)
    for idx in tqdm(similar_indexes):
        tr1 = sequences[idx]
        tr2 = generate_default_typo(tr1, vocab)
        if tr1 != tr2:
            dist = edit_distance(tr1, tr2)
            similar_examples.append((tr1, tr2, dist))

    examples = []
    examples.extend(dissimilar_examples)
    examples.extend(similar_examples)
    examples = pd.DataFrame(examples, columns=['sequence_a', 'sequence_b', 'distance'])

    train, test = train_test_split(examples, test_size=args.test_size)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

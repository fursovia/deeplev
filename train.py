import argparse
from itertools import chain
from pathlib import Path

import torch
import torch.optim as optim
from allennlp.common.util import dump_metrics
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.common import Params

from deeplev.dataset import LevenshteinReader
from deeplev.model import DeepLevenshtein
from deeplev.utils import load_weights

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=int, default=-1, help="cuda device number")
parser.add_argument("--model_dir", type=str, required=True, help="where to save checkpoints")
parser.add_argument("--data_dir", type=str, required=True, help="where train.csv and test.csv are")
parser.add_argument("--config", type=str, default="model_config/bilstm.jsonnet")

parser.add_argument("--num_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument(
    "--patience", type=int, default=2, help="Number of epochs to be patient before early stopping",
)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--lazy", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    data_path = Path(args.data_dir)

    reader = LevenshteinReader(lazy=args.lazy)
    train_dataset = reader.read(data_path / "train.csv")
    test_dataset = reader.read(data_path / "test.csv")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    dump_metrics(model_dir / "args.json", args.__dict__)

    if args.resume:
        vocab = Vocabulary.from_files(model_dir / "vocab")
    else:
        vocab = Vocabulary.from_instances(chain(train_dataset, test_dataset))
        vocab.save_to_files(model_dir / "vocab")

    iterator = BucketIterator(
        batch_size=args.batch_size, sorting_keys=[("sequence_a", "num_tokens"), ("sequence_b", "num_tokens")],
    )
    iterator.index_with(vocab)

    model = DeepLevenshtein.from_params(params=Params.from_file(args.config), vocab=vocab)
    if args.resume:
        load_weights(model, model_dir / "best.th")

    if args.cuda >= 0 and torch.cuda.is_available():
        model.cuda(args.cuda)

    # TRAINING
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=test_dataset,
        serialization_dir=model_dir,
        patience=args.patience,
        num_epochs=args.num_epochs,
        cuda_device=args.cuda,
    )

    results = trainer.train()

# Deep Levenshtein

# Dataset

Run this command to create a dataset

```bash
PYTHONPATH=. python scripts/dataset_from_csv.py \
    --csv-path ./data/dataset.csv \
    --col-name text \
    --output-dir ./data
```


# Training

To start a training process, run the following command

```bash
export GPU_ID="-1"
export TRAIN_DATA_PATH="./tests/fixtures/data/train.jsonl"
export VALID_DATA_PATH="./tests/fixtures/data/valid.jsonl"

allennlp train ./training_config/cnn.jsonnet \
    --serialization-dir ./logs/test \
    --include-package deeplev
```

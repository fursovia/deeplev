# Deep Levenshtein

Планы

* Обучить `rus/en` сетки для разных функций расстояний (Jaccard/Levenshtein/WER/BLEU/etc).
* Сравнить с популярными либами (https://github.com/ekzhu/datasketch, elastic, etc)
* Сделать полноценную либу с коллекцией предобученных моделек


# Datasets

How to create a dataset:
```bash
PYTHONPATH=. python scripts/create_dataset.py \
    --csv_path data/dblp/DBLP.csv \
    --col_name Title \
    --output_dir data/dblp/
```

* [Chunk of DBLP Dataset](https://www.kaggle.com/jakboss/chunk-of-dblp-dataset)


# Metrics

How to start the training

```bash
python train.py \
    --cuda 0 \
    --model_dir experiments/my_experiment \
    --data_dir data/dblp/
```


L1 error on the validation set

| Dataset\Model Name 	| Emb + LSTM |Emb + LSTM+Att| One-Hot + CNN | Emb + CNN |
|--------------------	|:----------:|:--------:	|:---------:	|:---------:| 
| Chunk of DBLP Dataset | 41.7181 	 |  34.6942 	|  12.4882     	|  10.9000  |
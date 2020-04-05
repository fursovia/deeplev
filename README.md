# Deep Levenshtein

Планы

* Обучить `rus/en` сетки для разных функций расстояний (Jaccard/Levenshtein/WER/BLEU/etc).
* Сравнить с популярными либами (https://github.com/ekzhu/datasketch, elastic, etc)
* Сделать полноценную либу с коллекцией предобученных моделек


# Metrics

How to create a dataset:
```bash
PYTHONPATH=. python scripts/create_dataset.py \
    --csv_path data/dblp/DBLP.csv \
    --col_name Title \
    --output_dir data/dblp/
```


| Dataset\Model Name 	|   LSTM  	| LSTM+Att 	| CNN 	|
|--------------------	|:-------:	|:--------:	|:---:	|
| [Chunk of DBLP Dataset](https://www.kaggle.com/jakboss/chunk-of-dblp-dataset)| 41.7181 	|  34.6942 	|  ?  	|
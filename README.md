# Deep Levenshtein

Планы

* Обучить `rus/en` сетки для разных функций расстояний (Jaccard/Levenshtein/WER/BLEU/etc).
* Сравнить с популярными либами (https://github.com/ekzhu/datasketch, elastic, etc)
* Сделать полноценную либу с коллекцией предобученных моделек


# Datasets

```bash
PYTHONPATH=. python scripts/create_dataset.py \
    --csv_path data/dblp/dataset.csv \
    --output_dir data/dblp/
```

* [Chunk of DBLP Dataset](https://www.kaggle.com/jakboss/chunk-of-dblp-dataset)

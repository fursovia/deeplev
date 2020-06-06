{
  "dataset_reader": {
    "type": "levenshtein",
    // DO NOT CHANGE token_indexers
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        // should be set to the maximum value of `ngram_filter_sizes`
        "token_min_padding_length": 7
      }
    },
    "tokenizer": {
      "type": "character"
    },
    "lazy": false
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
  "model": {
    "type": "deep_levenshtein",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 100
        }
      }
    },
    "seq2vec_encoder": {
      "type": "cnn",
      "embedding_dim": 100,
      "num_filters": 8,
      "ngram_filter_sizes": [
        3,
        5,
        7
      ]
    }
  },
  "data_loader": {
    "batch_size": 64
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 2,
    "cuda_device": std.parseInt(std.extVar("GPU_ID"))
  }
}

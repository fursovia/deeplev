{
    "seq2vec_encoder": {
        "type": "onehot_cnn",
        "ngram_filter_sizes": [
            4,
            5
        ],
        "num_filters": 32
    },
    "text_field_embedder": {
        "token_embedders": {
            "tokens": {
                "type": "onehot_encoder"
            }
        }
    }
}
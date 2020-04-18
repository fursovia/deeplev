{
    "seq2vec_encoder": {
        "type": "cnn",
        "embedding_dim": 64,
        "ngram_filter_sizes": [
            4,
            5
        ],
        "num_filters": 32
    },
    "text_field_embedder": {
        "token_embedders": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 64
            }
        }
    }
}
{
    "seq2vec_encoder": {
        "type": "lstm",
        "bidirectional": true,
        "hidden_size": 32,
        "input_size": 64,
        "num_layers": 1
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
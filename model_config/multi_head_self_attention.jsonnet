{
    "seq2seq_encoder": {
        "type": "multi_head_self_attention",
        "attention_dim": 32,
        "input_dim": 64,
        "num_heads": 4,
        "values_dim": 16
    },
    "seq2vec_encoder": {
        "type": "bag_of_embeddings",
        "averaged": true,
        "embedding_dim": 64
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
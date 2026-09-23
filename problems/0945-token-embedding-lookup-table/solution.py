import numpy as np

def token_embedding_lookup(vocab_size: int, embed_dim: int, token_ids: list, seed: int = 0) -> list:
    """
    Build a random embedding table of shape (vocab_size, embed_dim) using
    np.random.default_rng(seed).standard_normal(...), then return the rows
    corresponding to token_ids as a nested list.
    """
    rng = np.random.default_rng(seed)
    embedding_table = rng.standard_normal((vocab_size, embed_dim))
    
    # Index into the embedding table using the token_ids and convert to a nested list
    return embedding_table[token_ids].tolist()
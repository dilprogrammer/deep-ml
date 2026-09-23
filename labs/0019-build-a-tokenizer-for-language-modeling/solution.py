from collections import Counter

def train_tokenizer(corpus: list[str], vocab_size: int):
    """
    Trains a Byte-Pair Encoding (BPE) style subword tokenizer on the corpus.
    """
    # 1. Initialize vocabulary with all unique characters present in the corpus
    vocab_chars = sorted(list(set(c for name in corpus for c in name)))
    
    # Ensure special handling or cap vocab size
    # We will represent tokens as tuples of characters (or strings)
    # Start with base vocabulary of individual characters mapped to IDs
    stoi = {ch: i for i, ch in enumerate(vocab_chars[:vocab_size])}
    itos = {i: ch for ch, i in stoi.items()}
    
    # Convert corpus names into lists of character tuples
    splits = [tuple(name) for name in corpus]
    
    # Iteratively merge most frequent pairs until vocab_size is reached
    merges = {}
    while len(stoi) < vocab_size:
        # Count frequency of adjacent pairs
        stats = Counter()
        for split in splits:
            for pair in zip(split[:-1], split[1:]):
                stats[pair] += 1
        
        if not stats:
            break
            
        # Get the most frequent pair
        best_pair = stats.most_common(1)[0][0]
        
        # Create a new token for the merged pair
        new_token = best_pair[0] + best_pair[1]
        
        # Assign next available ID if within limit
        new_id = len(stoi)
        if new_id >= vocab_size:
            break
            
        stoi[new_token] = new_id
        itos[new_id] = new_token
        merges[best_pair] = new_token
        
        # Apply the merge across all splits in the corpus
        new_splits = []
        for split in splits:
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i+1]) == best_pair:
                    new_split.append(new_token)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits.append(tuple(new_split))
        splits = new_splits

    def encode(text: str) -> list[int]:
        # Tokenize text greedily using learned merges
        tokens = tuple(text)
        # Apply merges in the order they were learned
        while True:
            # Find all available pairs in tokens
            pairs = list(zip(tokens[:-1], tokens[1:]))
            if not pairs:
                break
            # Find the earliest merge that can be applied
            applicable = [p for p in pairs if p in merges]
            if not applicable:
                break
            # Pick the first merge (or priority based on training)
            best = applicable[0]
            replacement = merges[best]
            
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best:
                    new_tokens.append(replacement)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = tuple(new_tokens)
            
        # Map tokens to IDs (fallback to character level if an unknown token appears)
        ids = []
        for t in tokens:
            if t in stoi:
                ids.append(stoi[t])
            else:
                for ch in t:
                    ids.append(stoi.get(ch, 0))
        return ids

    def decode(ids: list[int]) -> str:
        # Map IDs back to tokens and join them
        return "".join(itos[i] for i in ids)

    return encode, decode

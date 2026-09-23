import re

def encode(text, vocab):
    pattern = r'([,.:;?_!"()\']|--|\s)'
    tokens = re.split(pattern, text)
    cleaned_tokens = [t.strip() for t in tokens if t and t.strip()]
    return [vocab[token] for token in cleaned_tokens]

def decode(ids, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # Map IDs back to tokens and join with single spaces
    tokens = [inv_vocab[i] for i in ids]
    text = " ".join(tokens)
    
    # Remove the space before specified punctuation characters
    pattern = r'\s+([,.?!"()\'])'
    return re.sub(pattern, r'\1', text)
    pass
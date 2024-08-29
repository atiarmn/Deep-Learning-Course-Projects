import numpy as np

class DataGenerator:
    def __init__(self, path):
        self.path = path
        
        with open(path) as f:
            data = f.read().lower()
        
        self.chars = list(set(data))
        
        self.char_to_idx = {ch: i for (i, ch) in enumerate(self.chars)}
        self.idx_to_char = {i: ch for (i, ch) in enumerate(self.chars)}
        
        self.vocab_size = len(self.chars)
        
        with open(path) as f:
            examples = f.readlines()
        self.examples = [x.lower().strip() for x in examples]
 
    def generate_example(self, idx):
        example_chars = self.examples[idx]
        
        example_char_idx = [self.char_to_idx[char] for char in example_chars]
        
        X = [self.char_to_idx['\n']] + example_char_idx
        Y = example_char_idx + [self.char_to_idx['\n']]
        
        return np.array(X), np.array(Y)

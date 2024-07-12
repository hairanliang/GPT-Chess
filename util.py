# Utility functions
from typing import List, Tuple
import torch

# Have example [a, b, c, d, e, EOS] --> input : [a, b, c, d, e], target: [b, c, d, e, EOS]

def produce_pairs(ts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = ts[:, :-1]
    targets = ts[:, 1:]

    return inputs, targets



# Function to add EOS token to end of training example. This should be applied before applying max_length padding.
def add_eos(ls: List) -> List:
    out = ls.append("<EOS>") # idk what this is supposed to be
    return out

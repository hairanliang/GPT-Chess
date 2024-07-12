import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformer import Decoder
from transformers import GPT2Tokenizer
from typing import List
from dataloader import ChessDataset
import chess
import itertools
from util import produce_pairs

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load and preprocess the data
file = "/Users/hairanliang/Documents/Chess.com/MyGames2012-02_fixed.pgn"
f = open(file)
game_list = []

while True:
    game = chess.pgn.read_game(f)
    if game is None:
        break  # end of file
    
    moves = game.board().variation_san(game.mainline_moves())
    game_list.append(moves)

lissy = tokenizer(game_list)['input_ids']
lissy = list(itertools.chain.from_iterable(lissy))
vocab_size = 50257

# Initialize model, dataset, and dataloader
model = Decoder(vocab_size=vocab_size, block_size=100, num_dec_blocks=1, emb_dim=256, model_dim=256, num_heads=2)
ds = ChessDataset(game_list, tokenizer, max_length=100)
batch_size = 25
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

# Initialize optimizer and loss function
optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)  # Lower learning rate
criterion = F.cross_entropy

# Add gradient clipping
clip_value = 1.0

# Training loop
num_epochs = 1000

for epoch in range(num_epochs):
    for i, (input_ids, mask) in enumerate(dataloader):
        optimizer.zero_grad()
        inputs, targets = produce_pairs(input_ids)
        outputs = model(inputs)
        outputs = outputs.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        mask = mask.reshape(-1)
        loss = criterion(outputs, targets, reduction='none')
        loss = (loss * mask).sum() / torch.sum(mask)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()

        print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

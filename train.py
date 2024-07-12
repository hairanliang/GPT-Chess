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
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# tokenizer.pad_token = tokenizer.eos_token

# Load and preprocess the data
# file = "/Users/hairanliang/Documents/Chess.com/MyGames2012-02_fixed.pgn"
# f = open(file)
# game_list = []

# while True:
#     game = chess.pgn.read_game(f)
#     if game is None:
#         break  # end of file
    
#     moves = game.board().variation_san(game.mainline_moves())
#     game_list.append(moves)

# lissy = tokenizer(game_list)['input_ids']
# lissy = list(itertools.chain.from_iterable(lissy))
# vocab_size = 50257

file = "/Users/hairanliang/Documents/Chess.com/all_games_hairan.pgn"
f = open(file)

game_list = []
counter = 0
token_dict = {} # This will act as the tokenizer. 
game_counter = 0
while True:   # In the future, can think about modularizing this somehow
    game = chess.pgn.read_game(f)
    if game is None:
        break  # end of file
    
    moves_list = []
    # board = chess.Board()
    board = game.board()  
    for move in game.mainline_moves():
        san_move = board.san(move)
        # print(san_move)
        moves_list.append(san_move)

        if san_move not in token_dict:
            token_dict[san_move] = counter
            counter += 1
        
        board.push(move)
    
    game_list.append(moves_list) # This should be a list of lists, where each list is one chess game in standard notation (SAN), each move is one element in the list, no more numbers
    game_counter += 1

token_dict['EOS'] = len(token_dict) # Adding in an EOS/padding token

vocab_size = len(token_dict)
print(f"number of games: {game_counter}")
print(f"tokenizer dict: {token_dict}")
print(f"vocab_size: {vocab_size}")

# Initialize model, dataset, and dataloader
model = Decoder(vocab_size=vocab_size, block_size=120, num_dec_blocks=6, emb_dim=512, model_dim=512, num_heads=2)
ds = ChessDataset(game_list, token_dict, max_length=120)
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


# Next step is to scale up. Getting loss down from 8.05 to 8.00, but likely need WAY more games. Right now only have 100 LOL. Time for the lichess dataset oh god...

# Might want to build a generate function as well.

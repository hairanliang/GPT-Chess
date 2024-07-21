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
import pickle

game_list = []
counter = 1
token_dict = {} # This will act as the tokenizer/encoder
reverse_dict = {} # This will act as the decoder from tokens to string/SAN 
game_counter = 0

with open('LichessElite/game_list.pkl', 'rb') as file:
    game_list = pickle.load(file)

with open('LichessElite/token_dict.pkl', 'rb') as file:
    token_dict = pickle.load(file)

with open('LichessElite/reverse_dict.pkl', 'rb') as file:
    reverse_dict = pickle.load(file)

with open('LichessElite/counter.pkl', 'rb') as file:
    game_counter = pickle.load(file)

# Set first token to 0/'BOS'
token_dict['BOS'] = 0 
reverse_dict[0] = 'BOS'

pad_token = len(token_dict)
token_dict['PAD'] = pad_token # Adding in a padding token

eos_token = len(token_dict)
token_dict['EOS'] = eos_token # Adding in an EOS token

vocab_size = len(token_dict)
print(f"number of games: {game_counter}")
print(f"tokenizer dict: {token_dict}")
print(f"decoder dict: {reverse_dict}")
print(f"vocab_size: {vocab_size}")

# Initialize model, dataset, and dataloader
model = Decoder(vocab_size=vocab_size, block_size=10, num_dec_blocks=6, emb_dim=512, model_dim=512, num_heads=2)
ds = ChessDataset(game_list, token_dict, max_length=10)
batch_size = 1
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

# At this point, model is trained. So, we can begin to generate (will make this a function later)

# starting_move = 'e4' # Begin generating from a starting token

def generate(model, max_game_length, starting_move=None):

    # Start counter at 1 because we are predicting position 1 right now, unless we are given a starting move, in which it gets incremented to 2
    counter = 1
    # Initialize game to be generated
    game = ['BOS']
    # Only add onto the game if 
    if starting_move != None:
        game.append(starting_move)
        counter += 1

    # We want to append PAD tokens until we reach the max_game_length
    while len(game) < max_game_length:
        game.append('PAD')
    
    # Now, convert strings to tokens using token_dict
        
    game = [token_dict[x] for x in game]
    game = torch.tensor(game)
    # Now, we just want to feed it to the model continuously, until the token we are predicting gives us EOS
    
    while counter < max_game_length and game[counter] != eos_token:
        output = model(game.unsqueeze(dim=0)).squeeze(dim=0)
        # print(output)
        # print(output.shape)
        # Now, we need to sample from the output for the index we are interested in (which is based on our counter)
        probs = output[counter]
        # print(probs) # The probabilities at that same index is the respective one for that token
        counter += 1
        chosen_token = torch.multinomial(probs, num_samples=1)
        chosen_token_prob = probs[chosen_token] # Wow, need a reverse dict mapping from numbers to respective SANs
        print(chosen_token_prob)
        print(chosen_token)
        print(f"Move chosen: {reverse_dict[chosen_token.item()]}")
    

generate(model, 10, starting_move=None)


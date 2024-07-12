from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import chess.pgn
import matplotlib.pyplot as plt
import numpy as np
from util import produce_pairs


# Design choice: have the dataloader call the utility function which then handles the input target pairs here

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token

class ChessDataset(Dataset):
    def __init__(self, games, tokenizer, max_length):
        self.games = games
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        # Remember, this is applied to one element of the batch, it doesn't see the other batch elements here
        game = self.games[idx]
        # Somewhere in between here have to apply the utility functions
        tokenized_game = self.tokenizer(game, padding='max_length', truncation=True, max_length=self.max_length + 1, return_tensors="pt") # This is a buggy fix for the later use of the produce_pairs

        # print(tokenized_game)
        
        input_ids = tokenized_game['input_ids'].squeeze(0) 
        loss_mask = tokenized_game['attention_mask'].squeeze(0) # This makes it so that the padded tokens have no effect on the loss
        loss_mask = loss_mask[:-1] # This is to correct for the fact that the 
        return input_ids, loss_mask # attention mask not only helps for attention but also will be useful for calculating loss to ignore padded tokens
    
if __name__ == '__main__':
    f = open("/Users/hairanliang/Documents/Chess.com/MyGames2012-02_fixed.pgn")

    game_list = []

    while True:
        game = chess.pgn.read_game(f)
        if game is None:
            break  # end of file
        
        moves = game.board().variation_san(game.mainline_moves())
        game_list.append(moves)
    
    game_lengths = [len(game) for game in game_list]

    # plt.hist(game_lengths, bins=50)
    # plt.xlabel('Game Length')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Game Lengths')
    # plt.show()

    max_length = int(np.percentile(game_lengths, 95))
    print(f"Chosen max_length: {max_length}")
    ds = ChessDataset(game_list, tokenizer, 10)

    # Checking all the dataset values to check for something fishy

    print(len(ds))
    for u in range(len(ds)):
        print(ds[u])

    # Aha, I've found some games that actually never started...
    

    # print(ds[0])

    # chessDataLoader = DataLoader(ds, batch_size=6, shuffle=True)

    # first = next(iter(chessDataLoader))
    # # print(first[0].shape)
    # print(first[1][2].shape)





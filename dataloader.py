from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import chess.pgn
import matplotlib.pyplot as plt
import numpy as np
from util import produce_pairs
import torch

# Design choice: have the dataloader call the utility function which then handles the input target pairs here

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# tokenizer.pad_token = tokenizer.eos_token


# What will be fed in is a list of moves, for which all that has to be done is convert them to the token through a simple for loop
class ChessDataset(Dataset):
    def __init__(self, games, token_dict, max_length):
        # print("initialized")
        self.games = games
        self.token_dict = token_dict # This will just be a dictionary mapping from chess move to integer(Long dtype)
        self.max_length = max_length

    def __len__(self):
        # print("inside len")
        # print(f"len: {len(self.games)}")
        return len(self.games)

    def __getitem__(self, idx):
        # print("inside getitem")
        # Remember, this is applied to one element of the batch, it doesn't see the other batch elements here
        game = self.games[idx].copy() # Have to make sure don't mutate it with each access, so just make a copy

        # Now, need to append the EOS token to the end of the tokenized game: if len() < max_length + 1, then add until it is max_length + 1
        if len(game) < self.max_length + 1:
            while len(game) < self.max_length + 1:
                game.append('EOS')
        elif len(game) > self.max_length + 1:
            game = game[:self.max_length+1]
        # Somewhere in between here have to apply the utility functions
        tokenized_game = self.tokenize(game, self.token_dict)

        # print(f"Tokenized game: {tokenized_game}")
        # print(f"shape of tokenized game: {tokenized_game.shape}")
        
        EOS_token = len(self.token_dict) - 1
        loss_mask = [0 if token == EOS_token else 1 for token in tokenized_game]
        # input_ids = tokenized_game['input_ids'].squeeze(0) 
        # loss_mask = tokenized_game['attention_mask'].squeeze(0) # This makes it so that the padded tokens have no effect on the loss
        loss_mask = torch.tensor(loss_mask[:-1]) # This is to correct for the fact that the 
        return tokenized_game, loss_mask # attention mask (encoder-only since we pad on the right anyway) not only helps for attention but also will be useful for calculating loss to ignore padded tokens
    
    def tokenize(self, game, token_dict):
        # print("inside tokenize")
        # game is a list of chess moves
        for i, move in enumerate(game):
            if move not in token_dict:
                print(f"Move '{move}' not found in token_dict!")
            game[i] = token_dict[move]
            # print("error is in return statement")
        return torch.tensor(game)
                        
    
if __name__ == '__main__':

    file = "/Users/hairanliang/Documents/Chess.com/MyGames2012-02_fixed.pgn"
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
    print(f"game list: {game_list}")


    ds = ChessDataset(game_list, token_dict, max_length=120)

    print(f"length of ds inside main: {len(ds)}")
    print(f"first ds element inside main{ds[0]}")
    print(ds[0])
    # for u in range(len(ds)):
    #     print(u)
    #     print(ds[u])
    # f = open("/Users/hairanliang/Documents/Chess.com/MyGames2012-02_fixed.pgn")

    # game_list = []

    # while True:
    #     game = chess.pgn.read_game(f)
    #     if game is None:
    #         break  # end of file
        
    #     moves = game.board().variation_san(game.mainline_moves())
    #     game_list.append(moves)
    
    # game_lengths = [len(game) for game in game_list]

    # # plt.hist(game_lengths, bins=50)
    # # plt.xlabel('Game Length')
    # # plt.ylabel('Frequency')
    # # plt.title('Distribution of Game Lengths')
    # # plt.show()

    # max_length = int(np.percentile(game_lengths, 95))
    # print(f"Chosen max_length: {max_length}")
    # ds = ChessDataset(game_list, tokenizer, 10)

    # # Checking all the dataset values to check for something fishy

    # print(len(ds))
    # for u in range(len(ds)):
    #     print(ds[u])

    # Aha, I've found some games that actually never started...
    

    # print(ds[0])

    # chessDataLoader = DataLoader(ds, batch_size=6, shuffle=True)

    # first = next(iter(chessDataLoader))
    # # print(first[0].shape)
    # print(first[1][2].shape)





import chess.pgn
import pickle 

# Need to add BOS token, then prepend it to all of the games.
# Append EOS to the end of each game, before adding padding token!
# Need to add GPU/Cuda stuff

file = "/Users/hairanliang/Downloads/lichess_elite_2021-12.pgn"
f = open(file)

game_list = []
counter = 1
token_dict = {} # This will act as the tokenizer/encoder
reverse_dict = {} # This will act as the decoder from tokens to string/SAN 
game_counter = 0

while True:   # In the future, MODULARIZE this into a tokenizer.py file. Tokenize everything, save into file, then read that file instead for training.
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
            reverse_dict[counter] = san_move
            counter += 1
        
        board.push(move)
    
    game_list.append(moves_list) # This should be a list of lists, where each list is one chess game in standard notation (SAN), each move is one element in the list, no more numbers
    game_counter += 1

with open("LichessElite/game_list.pkl", "wb") as file:
    pickle.dump(game_list, file)

with open("LichessElite/token_dict.pkl", 'wb') as file:
    pickle.dump(token_dict, file)

with open("LichessElite/reverse_dict.pkl", 'wb') as file:
    pickle.dump(reverse_dict, file)

with open("LichessElite/counter.pkl", 'wb') as file:
    pickle.dump(game_counter, file)

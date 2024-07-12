# Tokenizer time happy

import chess.pgn

# For chess, just need to get all the possible moves

# Can just grab all the possible moves in the dataset

file = "/Users/hairanliang/Documents/Chess.com/MyGames2012-02_fixed.pgn"
f = open(file)

game_list = []
counter = 0
token_dict = {}
game_counter = 0
while True:
    
    
    game = chess.pgn.read_game(f)
    if game is None:
        break  # end of file
    if game_counter == 0:
        print(game)
    
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
    
    game_list.append(moves_list)
    game_counter += 1
print(token_dict)

# Hm, I just want to turn each game into 
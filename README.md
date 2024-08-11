
# GPT-Chess

I built and trained a GPT model that can generate chess games in PGN notation from scratch in PyTorch. I attached it to an API using FastAPI and containerized it using Docker Compose.



## Overview

Generative pre-trained transformer, or GPT, models have taken the world by storm. ChatGPT and similar LLMs are being used by students, parents, and enterprises across the world to solve a variety of problems. While I knew the basic theory of transformers, I wanted to delve deeper and actually create one from scratch by following the original 
[AttentionIsAllYouNeed](https://arxiv.org/abs/1706.03762) paper. 

Having played professional chess for close to half of my life, I wanted to tie the project to chess somehow, which is how I came up with the idea of GPT-Chess. Many of us may be familiar with the famous case of AlphaZero that dominated the chess scene, not only beating the best players but more importantly the strongest chess engines. My goal was not to dethrone AlphaZero——my hunch is that reinforcement learning based models will always be the king of chess engines——but to create a GPT model that could produce reasonable chess games.

With this project, I wanted to learn how to code up the Transformer architecture, and consequently the GPT-2 architecture all in PyTorch. I wanted to learn more about training enormous models by firsthand training one——my base model ended up with 75,222,251 parameters! Lastly, I wanted some more practical experience deploying deep learning models, so I experimented with FastAPI and Docker to containerize and wrap my model with an API that others can easily call and use to generate chess games, whether on a virtual machine or local machine.



## How do we convert chess games to text?

Luckily for me, chess games have an easy way to be converted to text as it is written into professional chess itself——chess PGN notation. In formal games, chess players are responsible for writing down each move made in the game. This serves two purposes: 1. If there is a disagreement later in the game and a referee needs to be involved, the players can refer to the game notation 2. So that players can replay their games and learn from their mistakes/admire their victories (the former is less fun but infinitely more helpful for improvement...).

Here is an example of chess notation for the famous 4 move checkmate: 

1\. e4, e5 2. Bc4, Nc6 3. Qh5, Nf6 4. Qxf7# 

![mate_in_4](https://github.com/user-attachments/assets/4a05c5ac-4c60-420e-a72e-301d5e192409)

From the notation, it hints at how we can tokenize these chess games for our transformer model. 

## Tokenizer and Dataloader

At the end of the day, a chess game is just a sequence of moves. Thus, I tokenized each game by representing each move by a token. Thus, for each dataset of chess games, the total number of tokens is just the number of distinct moves. This varied based on the dataset, but for example, the number of tokens for a smaller dataset I used (Black Knight Tango) was only around 3000, whereas for the larger dataset, it was about 9000 tokens.

It's also important to note that besides the tokens for each unique chess move in the dataset, we also need special tokens for training. These tokens are the 'BOS' (Beginning of Sequence), 'EOS' (End of Sequence), and 'PAD' (Padding) tokens. 'BOS' is self-explanatory: it will be added to the beginning of each tokenized chess game in the dataloader: 

    def __getitem__(self, idx):
        
        prepend = 'BOS'
        
        # Have to make sure don't mutate it with each access, so just make a copy
        game = self.game_list[idx].copy() 
        
        game = [prepend] + game

This way, during training, when the model sees the 'BOS' token, it knows to then predict the first move of the chess game.

'EOS' tokens are also straight forward——they will be added to the end of each game so that the model knows that the game is over, signalling no more tokens coming afterwards.

The 'PAD' special token is very important in terms of batching and training. Within each batch, there could be different length games, which would be a huge problem if we want efficient training. So, to make things easier, we can simply pad every game to a fixed amount, such as 80 tokens——about the average length of a chess game (40 moves for each side). 

So, for every game in our dataset, if it is shorter than the max game length, we pad it to the max game length and append the 'EOS' token. If it's longer than the max length, we shrink the game and then append the 'EOS' token. 

        if len(game) >= self.max_game_length:
            game = game[:self.max_game_length - 1]
            game.append('EOS')
        else:
            game.append('EOS')
            while len(game) < self.max_game_length:
                game.append('PAD')

We then have to tokenize the games, and provide a mask that the training function will use to ensure that 'PAD' tokens do not affect the loss calculations and consequently backpropagation of the model's weights.

        tokenized_game = self.tokenize(game, self.token_dict)

        PAD_token = len(self.token_dict) - 2 # PAD token is second last token
        EOS_token = len(self.token_dict) - 1 # EOS token is last token
        loss_mask = torch.tensor([0 if token == PAD_token else 1 for token in tokenized_game])

        loss_mask = loss_mask[:-1] # This is to correct for the fact that we grabbed max_length + 1 tokens, since we are using the produce_pairs function

        return tokenized_game, loss_mask

Feel free to check out the dataloader.py file to see the actual tokenize function defined, and other details left out here.

Side note: In the future, it is also possible to incorporate dynamic padding, where instead of padding to a max game length for all games, we can do it batch-wise, padding every game to the longest length in that particular batch. This could speed up training because we aren't always padding games to a fixed number, and also allow for cases where we want to learn from games that are longer than 80 tokens for example.


## Datasets

Until now, I have two datasets that the model has been trained on. The first one is small dataset of 3915 games, all in one particular chess opening, the [Black Knight Tango](https://www.pgnmentor.com/files.html#openings). The purpose of this dataset was to see if the model could overfit to the first several moves, since all the games in this dataset share the first 4 chess moves, that is, 2 moves from white and 2 moves from black: 1.d4 Nf6 2.c4 Nc6. Fortunately, the model successfully produced games that overfit to the opening moves——here is one amusing example: 

['BOS', 'd4', 'Nf6', 'c4', 'Nc6', 'Nf3', 'd6', 'Nc3', 'e5', 'd5', 'Ne7', 'e4', 'g6', 'Be2', 'Be7', 'O-O', 'O-O', 'a6', 'Qc2', 'Bf5', 'f4', 'c5', 'g3', 'Qf6', 'cxd6', 'cxd6', 'cxd6', 'cxd6', 'cxd6', 'cxd6', 'cxd6', 'cxd6', 'g3', 'd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'EOS']

Clearly, the model learns the first several moves as the games, but soon, it quickly makes illegal moves, the first being 'Be7'——the Black bishop moves to a square the black knight is already on, as shown by the earlier 'Ne7.' We can also see that it starts to hallucinate many captures of 'exd5' and 'cxd5', a sign that our model is receiving nowhere close to the amount of training data it needs to be decent.

The second dataset, and the main one that I am interested in training the model for, is an enormous dataset of ~500,000 games between "Elite" players, which is defined by players with ratings greater than 2300. Specifically, I chose the December 2021 [dataset](https://database.nikonoel.fr/)——thank you @NikoNoel for this!

In the future, once I'm able to finish training the second dataset, I want to use an even bigger dataset, with at least 1 million games. The reason for this is that Transformers are very data hungry, which is a perfect segueway to the next section.

## Model Architecture

The GPT architecture may seem overwhelming at first, but we'll build from the ground up, starting with the original transformer. I replicated the transformer architecture from the AttentionIsAllYouNeed paper, with a few changes I will talk about later. Here is a quick overview from the original paper itself (Vaswani et al., 2017):

![transformer](https://github.com/user-attachments/assets/44727288-9fa3-46cb-9f09-aad00895f75a)

As can be seen, there are two parts to the Transformer Architecture——the encoder (left) and decoder (right). The original purpose of the Transformer was for machine translation, so the encoder would deal with encoding a sentence in french for example, pass this encoding to the decoder, which would then produce the translation of that sentence in english. Furthermore, the Decoder would interact with the Encoder through Cross-Attention, but in my implementation, I did not need it so it is not discussed here.

Since I am only trying to generate chess games, and emulating GPT-2, I actually only need to use the Decoder, AKA a Decoder-only Transformer (which is what Chat-GPT is). However, in the future, I can consider using an encoder to encode the current chess position, which could potentially help the decoder generate even better games. For that reason, I implemented both the Encoder and Decoder and will talk about both.

We will start with the first step, going from tokens——which we got from the tokenizer——to embeddings. We want to represent each unique token (AKA chess move) with a vector that is learned through training, and we do this through a simple PyTorch Embedding table:

    class TokenEmbedding(nn.Module):

        """
        Class that goes from raw tokens to embedding dimension.
        """

        def __init__(self, vocab_size, emb_dim):
            super(TokenEmbedding, self).__init__()
            self.vocab_size = vocab_size
            self.emb_dim = emb_dim
            self.token_embedding = nn.Embedding(self.vocab_size, self.emb_dim)
            
        def forward(self, x):
            out = self.token_embedding(x)
            return out

Now, we have a way to represent each possible chess move with an embedding, however it's clear that playing a move like d4 on the first move of the game is very different from playing d4 on move 20, so how can our transformer model capture differences in token position? We generate a Positional Embedding for each token.

    class PositionalEmbedding(nn.Module):

        """
        Class that provides embedding that captures the positional information of a token.
        """

        def __init__(self, block_size, emb_dim):
            super(PositionalEmbedding, self).__init__()
            self.block_size = block_size
            self.emb_dim = emb_dim
        
        def forward(self, x):
            # Use torch.no_grad to make sure PositionalEmbedding not part of computational graph
            with torch.no_grad():
                # Credit to https://github.com/hyunwoongko/transformer 

                pos_embedding = torch.zeros(self.block_size, self.emb_dim)
                position = torch.arange(self.block_size).unsqueeze(dim=1)
                _2i = torch.arange(0, self.emb_dim, step=2)
                pos_embedding[:, 0::2] = torch.sin(position / (10000 ** (_2i / self.emb_dim)))
                pos_embedding[:, 1::2] = torch.cos(position / (10000 ** (_2i / self.emb_dim)))
           
            return pos_embedding

The original AttentionIsAllYouNeed paper covered two possible ways of doing the PositionalEmbedding——learned (using neural networks) and the one I used, sinusoidal positional embeddings (not learned). I have to give credit to @hyunwoongko for his implementation of the sinusoidal positional embedding, which was tricky to wrap my head around at first. Thank you to [Amirhossein Kazemnejad](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) and [Mehreen Saeed's](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/) blogs for helping me understand this crucial concept——feel free to check them out for more information. 

To combine both the positional and token embedding information for each token, we just add the two together:

    x = self.tok_emb(x) + self.pos_emb(x)

Now that we have the token representations, they are ready to be passed into the Encoder/Decoder. Let's start with the Encoder first, but keep in mind that the components of the Encoder and Decoder are very similar. 

### LayerNorm

LayerNorm is a method to normalize the features of each token independently. This helps stabilize training, and in the original AttentionIsAllYouNeed paper, they applied the LayerNorm after the Attention layer (we will get to what Attention is shortly). However, after the [PreNorm paper](https://arxiv.org/pdf/2002.04745), the PreNorm method has been favored, so I applied LayerNorm right after the Embeddings and before Attention. Thank you Andrej Karpathy and specifically this [video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5878s) for being a great guide for PreNorm and the overall model building process. 

### Attention

At the heart of the Transformer architecture is the attention mechanism, which allows the tokens to "talk with each other." In other words, attention allows tokens from any position to incorporate their information with tokens in other positions in the same sequence. 

There are many amazing explanations of attention on the Internet or in college courses, but I'll include a brief, intuitive way of how I understand it, having learned it from so many different people. For sake of clarity, let's leave the chess world for a moment and pretend we are calculating attention on a english sentence like: 

Cherries are Hairan's favorite fruit 

It's clear to anyone reading the sentence that the words "Cherries" and "fruit" along with "Hairan's" and "favorite" are very related to each other in this phrase. Ideally, when we train our model——let's say it was a GPT model——hopefully it would learn these relations so that when it generates sentences after this one, it can make use of these connections to create more coherent and logical sentences. But, how does our GPT model learn these relations? That's where Queries, Keys, and Values——the foundations of attention——come into play.

Queries: Think of these as a token asking for what it's looking for, similar to how you might query a database in SQL. Each token will have a unique query, which will depend on its own characteristics. This is as far as I can explain it, and it is very high-level, which is mostly because interpretabililty around Transformers is an ongoing research problem——we don't know why they are so powerful. 

To get the queries for each token, we simply use a PyTorch linear layer that goes from the original embeddings we got after the positional + token embeddings.

    self.Q = nn.Linear(model_dim, model_dim)



## Training

**Still training on the largest dataset with the help of GPUs, so this section will be updated once done with training.

## FastAPI

## Docker 




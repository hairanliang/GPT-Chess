
## Overview

Generative pre-trained transformer, or GPT, models have taken the world by storm. ChatGPT and similar LLMs are being used by students, parents, and enterprises across the world to solve a variety of problems. While I knew the basic theory of transformers, I wanted to delve deeper and actually create one from scratch by following the original 
[AttentionIsAllYouNeed](https://arxiv.org/abs/1706.03762) paper. 

Having played professional chess for close to half of my life, I wanted to tie the project to chess somehow, which is how I came up with the idea of GPT-Chess. Many of us may be familiar with the famous case of AlphaZero that dominated the chess scene, not only beating the best players but more importantly the strongest chess engines. My goal was not to dethrone AlphaZero——my hunch is that reinforcement learning based models will always be the king of chess engines.

With this project, I wanted to learn how to code up the Transformer architecture, and consequently the GPT-2 architecture all in PyTorch. I wanted to learn by firsthand training such large models——my base model ended up with 75,222,251 parameters! Lastly, I wanted some more practical experience deploying such large models, so I experimented with FastAPI and Docker Compose to containerize and wrap my model with an API that others can easily call. 

## How do we convert chess games to text?

Luckily for me, chess games have an easy way to be converted to text as it is written into professional chess itself——chess PGN notation. In formal games, chess players are responsible for writing down each move made in the game. This serves two purposes: 1. If there is a disagreement later in the game and a referee needs to be involved, the players can refer to the game notation 2. So that players can replay their games and learn from their mistakes/admire their victories (the former is less fun but infinitely more helpful for improvement...).

Here is an example of chess notation for the famous 4 move checkmate: 

1\. e4, e5 2. Bc4, Nc6 3. Qh5, Nf6 4. Qxf7# 

![mate_in_4](https://github.com/user-attachments/assets/4a05c5ac-4c60-420e-a72e-301d5e192409)

From the notation, it hints at how we can tokenize these chess games for our transformer model. 

## Tokenizer

At the end of the day, a chess game is just a sequence of moves. Thus, I tokenized each game by representing each move by a token. Thus, for each dataset of chess games, the total number of tokens is just the number of distinct moves. This varied based on the dataset, but for example

## Datasets

Until now, I have two datasets that the model has been trained on. The first one is small dataset of 3915 games, all in one particular chess opening, the [Black Knight Tango](https://www.pgnmentor.com/files.html#openings). The purpose of this dataset was to see if the model could overfit to the first several moves, since all the games in this dataset share the first 4 chess moves, that is, 2 moves from white and 2 moves from black: 1.d4 Nf6 2.c4 Nc6. Fortunately, the model successfully produced games that overfit to the opening moves——here is one amusing example: 

['BOS', 'd4', 'Nf6', 'c4', 'Nc6', 'Nf3', 'd6', 'Nc3', 'e5', 'd5', 'Ne7', 'e4', 'g6', 'Be2', 'Be7', 'O-O', 'O-O', 'a6', 'Qc2', 'Bf5', 'f4', 'c5', 'g3', 'Qf6', 'cxd6', 'cxd6', 'cxd6', 'cxd6', 'cxd6', 'cxd6', 'cxd6', 'cxd6', 'g3', 'd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'EOS']

Clearly, the model learns the first several moves as the games, but soon, it quickly makes illegal moves, the first being 'Be7'——the Black bishop moves to a square the black knight is already on, as shown by the earlier 'Ne7.'

The second dataset, and the main one that I am interested in training the model for, is an enormous dataset of ~500,000 games between "Elite" players, which is defined by players with ratings greater than 2300. Specifically, I chose the December 2021 [dataset](https://database.nikonoel.fr/)——thank you @NikoNoel for this!

In the future, once I'm able to finish training the second dataset, I want to use an even bigger dataset, with at least 1 million games. The reason for this is that Transformers are very data hungry, which is a perfect segueway to the next section.

## Model Architecture

The GPT architecture may seem overwhelming at first, but we'll build from the ground up, starting with the original transformer. I replicated the transformer architecture from the AttentionIsAllYouNeed paper, with a few changes I will talk about later. Here is a quick overview from the original paper itself (Vaswani et al., 2017):

![transformer](https://github.com/user-attachments/assets/44727288-9fa3-46cb-9f09-aad00895f75a)

As can be seen, there are two parts to the Transformer Architecture——the encoder and decoder. 


## Training

**Still training on the largest dataset, so this section will be updated once done so


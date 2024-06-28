# I don't want to do the tokenizer first...
import torch.nn as nn
import torch

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
    
## Use torch.no_grad to make sure PositionalEmbedding not part of computational graph
class PositionalEmbedding(nn.Module):
    """
    Class that provides embedding that captures the positional information of a token.
    """
    def __init__(self, block_size, emb_dim):
        super(PositionalEmbedding, self).__init__()
        self.block_size = block_size
        self.emb_dim = emb_dim
    
    def forward(self, x):
        # Implement the sine + cosine
        with torch.no_grad():
            pos_embedding = torch.zeros(self.block_size, self.emb_dim)
            position = torch.arange(self.block_size).unsqueeze(dim=1)
            _2i = torch.arange(0, self.emb_dim, step=2)
            pos_embedding[:, 0::2] = torch.sin(torch.tensor(position / (10000 ** (_2i / self.emb_dim))))
            pos_embedding[:, 1::2] = torch.cos(torch.tensor(position / (10000 ** (_2i / self.emb_dim))))
        return pos_embedding

class MultiHeadAttention(nn.Module):

    """
    Class that performs the splitting of Q, K, V into separate attention calculations, concatenating at the end.
    """

    def __init__(self, emb_dim, model_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.Q = nn.Linear(emb_dim, model_dim) #  These are the weights
        self.K = nn.Linear(emb_dim, model_dim) 
        self.V = nn.Linear(emb_dim, model_dim)
        self.attention = ScaledDotProductionAttention()

    def forward(self, x):
        print(f"Input before attention{x.shape}")
        q, k, v = self.split(self.Q(x)), self.split(self.K(x)), self.split(self.V(x)) # queries, keys, values
        output, k, v = self.attention(q, k, v, self.model_dim)
        output = self.concatenate(output)
        # Need to pass through w_o here, which should be (h*c_heads, model_dim)
        return output, k, v

    # Go from (B, T, C) --> (B, h, T, C // h). There's other ways of doing this, but people have found this is efficient
    def split(self, x):
        print(f"pre-split {x.shape}")
        B, T, C = x.shape
        out = x.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # Intuitively, each channel's/head's calculations are separate from each other, just like different channels in a CNN
        return out
    
    def concatenate(self, x):
        B, n_heads, T, T = x.shape
        print(f"Before concatenate: {x.shape}") # It's (B, n_heads, T, T)
        out = x.transpose(1,2).contiguous().view(B, T, T * n_heads) # Want to be (B, T, T*n_heads)
        print(f"After concatenate: {out.shape}")
        return out
    
## I see, so don't necessarily need to have the q, k, v defined in here, this can just serve as a function
class ScaledDotProductionAttention(nn.Module): 
    """
    Class that performs the attention calculation using Q, K, V.
    """
    def __init__(self):
        super(ScaledDotProductionAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1) # Want to be along rows
    
    # Assume q, k, v are the tensors, already multiplied by the Q, K, V weights. They should also be of shape (B, T, C/h, h)
    def forward(self, q, k, v, model_dim):
        pre_softmax = (q @ k.transpose(-1, -2)) / (model_dim ** 0.5) 
        post_softmax = self.softmax(pre_softmax) 
        out = post_softmax @ v
        print(f"Inside Attention: output shape after split before concatenate: {out.shape}")
        return out, k, v # Do I want to return more than just out?

class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-forward Network. 
    """
    def __init__(self, model_dim):
        super(FeedForwardNetwork, self).__init__()
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
    
class EncoderBlock(nn.Module):
    """
    Encoder block made up of a self-attention, layer norm, residual connection, and FFN.
    """
    def __init__(self, emb_dim, model_dim, num_heads):
        super(EncoderBlock, self).__init__()
        # don't need the embeddings here because they only get used once
        self.emb_dim = emb_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        
        self.mha = MultiHeadAttention(emb_dim, model_dim, num_heads) # q, k, v all defined within here
        self.ffn = FeedForwardNetwork(model_dim)
        
    
    def forward(self, x):
        # Just getting the output for now from prev. encoder block
        # print(x.shape)
        # Define the layer-norm here because first encoder block's first layer norm has emb_dim, but all the other layer norms have model_dim
        ln1_dim = x.shape[-1]
        ln_1 = nn.LayerNorm(ln1_dim)
        x = ln_1(x) # Using pre-norm method due to increased performance
        # print(x.shape)
        attention, k, v = self.mha(x)
        print(attention.shape)
        ln2_dim = attention.shape[-1]
        ln_2 = nn.LayerNorm(ln2_dim)
        print(attention.shape)
        attention = ln_2(attention)
        print(attention.shape)
        output = self.ffn(attention)
        output += attention # residual connection applied after the output

        return output # Only the last encoder will need the k, v, but the rest need attention to pass as input to next encoder. Have to code in receiving k, v?


class Encoder(nn.Module):
    """
    Encoder class that makes up the Encoder side of the network
    """
    def __init__(self, vocab_size, block_size, num_blocks, emb_dim, model_dim, num_heads):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEmbedding(block_size, emb_dim)
        self.tok_emb = TokenEmbedding(vocab_size, emb_dim)
        self.encoders = nn.ModuleList([EncoderBlock(emb_dim, model_dim, num_heads) for i in range(num_blocks)])
        
    def forward(self, x):
        x = self.tok_emb(x) + self.pos_emb(x) # Applying the token and positional embeddings to get input for first encoder block
        print(f"Shape after embedding: {x.shape}")
        for i, l in enumerate(self.encoders):
            print(f"On this encoder block: {i}")
            x = self.encoders[i](x)
        return x # are we done? pogchampion???
    
if __name__ == '__main__':
    # Let's imagine the vocab_size is 20

    input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]) # These are the tokens, so dimension of (batch, time)
    B, T = input.shape
    print(f"input shape {input.shape}")
    encoder = Encoder(vocab_size=20, block_size=T, num_blocks=6, emb_dim=8, model_dim=8, num_heads=2)
    output = encoder(input)
    print(output)

    


        





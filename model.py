import torch
import torch.nn as nn

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_embedding = nn.Embedding(vocab_size, embedding_dim)
    
    # Define x to be have a shape of [batch_size, 2]
    def forward(self, x):
        embed_input = self.input_embedding(x.T[0])
        embed_output = self.output_embedding(x.T[1])
        return (embed_input*embed_output).sum(dim=1)
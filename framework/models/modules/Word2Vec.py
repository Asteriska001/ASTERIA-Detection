import torch
import torch.nn as nn
import torch.nn.functional as F

#the impl of word2vec: CBOW and SkipGram

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_size, context_size):
        super(CBOW, self).__init__()
        
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.linear1 = nn.Linear(embed_size, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGram, self).__init__()
        
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = self.linear(embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

# Example
# model = SkipGram(vocab_size=len(vocab), embed_size=100)

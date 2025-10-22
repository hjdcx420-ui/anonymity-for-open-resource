import torch.nn as nn
import torch.nn.functional as F
import torch

class MultiHeadAttentionAd(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MultiHeadAttentionAd, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.q_layer = nn.Linear(in_dim, out_dim)
        self.k_layer = nn.Linear(in_dim, out_dim)
        self.v_layer = nn.Linear(in_dim, out_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        Q = self.q_layer(q)
        K = self.k_layer(k)
        V = self.v_layer(v)
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.out_dim ** 0.5)
        if mask is not None:
            attention_scores.masked_fill_(mask, -1e9)
        attention_weights = self.softmax(attention_scores)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

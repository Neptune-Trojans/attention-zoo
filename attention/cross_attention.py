import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.outputs import AttentionOutput


class CrossAttention(nn.Module):
    def __init__(self, query_dim, kv_dim):
        super(CrossAttention, self).__init__()
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.query = nn.Linear(query_dim, query_dim, bias=False)
        self.key   = nn.Linear(kv_dim,    query_dim, bias=False)
        self.value = nn.Linear(kv_dim,    query_dim, bias=False)


    def forward(self, x_q, x_k, x_v):
        queries = self.query(x_q)
        keys = self.key(x_k)
        values = self.value(x_v)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (keys.size(-1) ** 0.5)
        attention = F.softmax(scores, dim=-1)
        weighted = torch.matmul(attention, values)
        return AttentionOutput(output=weighted, attention_weights=attention)

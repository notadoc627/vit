import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from DotProductAtten import DotProductAtten as D

class MultiHeadAttention(nn.Module):
    def __init__(self, q, k, v, heads, num_hiddens, dropout = 0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.attention = D(dropout)
        self.heads = heads
        self.W_q = nn.Linear(q, num_hiddens)
        self.W_k = nn.Linear(k, num_hiddens)
        self.W_v = nn.Linear(v, num_hiddens)
        self.W_out = nn.Linear(num_hiddens, num_hiddens)

    def transpose_qkv(self, X, heads):
        # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
        # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
        X = X.reshape(X.shape[0], X.shape[1], heads, -1)
        X = X.permute(0, 2, 1, 3)
        # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
        # num_hiddens/num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X, heads):
        # (batch, heads, lens, /)
        X = X.reshape(-1, heads, X.shape[1], X.shape[2])
        # (batch, lens, heads, /)
        X = X.permute(0, 2, 1, 3)
        # (batch, lens, all_features)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, q, k, v, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        q = self.transpose_qkv(self.W_q(q), self.heads) # 8 196 96
        k = self.transpose_qkv(self.W_k(k), self.heads)
        v = self.transpose_qkv(self.W_v(v), self.heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.heads, dim = 0)
        # 每个head各算各 8 196 96
        output = self.attention(q, k, v, valid_lens)
        # print('output shape', ouptut.shape)
        # 1 196 768
        output_concat = self.transpose_output(output, self.heads)
        return self.W_out(output_concat)

batch_size = 1
pairs_num = 196
num_hiddens = 768

# 使用 torch.randn 生成具有随机值的张量
q = torch.randn(batch_size, pairs_num, num_hiddens)
k = torch.randn(batch_size, pairs_num, num_hiddens)
v = torch.randn(batch_size, pairs_num, num_hiddens)

m = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, 8, num_hiddens)
m.forward(q, k, v, None)

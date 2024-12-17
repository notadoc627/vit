import torch
import torch.nn as nn
import math
import numpy as np
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 待填充矩阵
        pe = torch.zeros(max_len, num_hiddens)
        # 对于每一行
        for pos in range(max_len):
            # 对于每一列
            for i in range(0, num_hiddens, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i) / num_hiddens))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * i) / num_hiddens))
        pe.unsqueeze(0)

    def forward(self, X):
        seq_len  = X.size(1)
        X = X + self.pe[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

import torch
import torch.nn as nn

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.layernormal = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.layernormal(self.dropout(Y) + X)
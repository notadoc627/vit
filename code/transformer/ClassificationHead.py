import torch
import torch.nn as nn
from einops.layers.torch import Reduce

class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super(ClassificationHead, self).__init__()
        # 定义各个子模块
        self.reduce = Reduce('b n e -> b e', reduction='mean')
        self.layer_norm = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        # 按顺序应用各个子模块
        x = self.reduce(x)
        x = self.layer_norm(x)
        x = self.linear(x)
        return x


import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat
'''
 分patch+cls token+positional encoding
'''
class PatchesEmbedding(nn.Module):
    def __init__(self, img_size, in_channels, patch_size, embedding_size, **kwargs):
        super(PatchesEmbedding, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            # 对每一个patch
            nn.Conv2d(in_channels, embedding_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
            # Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size), # 裁剪+拉平
            # nn.Linear(patch_size * patch_size * in_channels, embedding_size)
        )
        # （1， 1， 768）
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size)**2 + 1, embedding_size))

    def forward(self, X):
        b, _, _, _= X.shape
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b = b)
        X = self.projection(X)
        X = torch.cat([cls_tokens, X], dim=1)
        X += self.positions
        return X

# print(PatchesEmbedding(224, 3, 16, 768)(torch.ones([1, 3, 224, 224])).shape)
import torch
import torch.nn as nn
from einops import rearrange
from code.PatchesEmbedding import PatchesEmbedding
import math

class MultiHeadsAtten_vit(nn.Module):
    def __init__(self, q_size, k_size, v_size, heads, num_hiddens, dropout=0.1, **kwargs):
        super(MultiHeadsAtten_vit, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.heads = heads
        self.W_q = nn.Linear(q_size, num_hiddens)
        self.W_k = nn.Linear(k_size, num_hiddens)
        self.W_v = nn.Linear(v_size, num_hiddens)
        self.W_out = nn.Linear(num_hiddens, num_hiddens)
        self.dropout = nn.Dropout(dropout)

    def transpose_output(self, X, heads):
        # (batch, heads, lens, /)
        X = X.reshape(-1, heads, X.shape[1], X.shape[2])
        # (batch, lens, heads, /)
        X = X.permute(0, 2, 1, 3)
        # (batch, lens, all_features)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def masked_softmax(self, X, valid_lens= None):
        # X：3d valid_lens: 1d/2d
        # 如果是 2D，表示每个序列中每个位置的有效长度，需要将其展平。
        if valid_lens is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape = X.shape
            if valid_lens.dim() == 1:
                valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            else:
                valid_lens = valid_lens.reshape(-1)
            X = self.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=1e-6)
            return nn.Softmax(X.reshape(shape), dim=-1)

    def forward(self, X, valid_lens=None):
        q = rearrange(self.W_q(X), 'b n (h d) -> (b h) n d', h=self.heads)
        k = rearrange(self.W_k(X), 'b n (h d) -> (b h) n d', h=self.heads)
        v = rearrange(self.W_v(X), 'b n (h d) -> (b h) n d', h=self.heads)
        print("V shape: ", v.shape)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        print("score shape: ", scores.shape)
        self.attention_weights = self.masked_softmax(scores, valid_lens)
        print("attention shape: ", self.attention_weights.shape)
        output = torch.bmm(self.dropout(self.attention_weights), v)
        print("output shape: ", output.shape)
        output_concat = self.transpose_output(output, self.heads)
        return self.W_out(output_concat)

# batch_size = 2
# nums = 196
# num_hiddens = 768
#
# # 使用 torch.randn 生成具有随机值的张量
# q = torch.randn(batch_size, nums, num_hiddens)
# k = torch.randn(batch_size, nums, num_hiddens)
# v = torch.randn(batch_size, nums, num_hiddens)
# patches_embedded = PatchesEmbedding(224, 3, 16, 768)(torch.ones([1, 3, 224, 224]))
# print("patches_embedding's shape: ", patches_embedded.shape)
# print(MultiHeadsAtten_vit(768, 768, 768, 8, 768)(patches_embedded).shape)
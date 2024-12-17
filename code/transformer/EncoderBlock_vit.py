import torch
import torch.nn as nn
from code.PatchesEmbedding import PatchesEmbedding
from code.transformer.AddNorm import AddNorm
from code.transformer.FFN import FFN
from code.transformer.MultiHeadsAtten_vit import MultiHeadsAtten_vit

# 每一个block里面包含ffn和多头注意力，他们之间使用addnorm和残差连接
class EncoderBlock_vit(nn.Module):
    # 所以输入是ffn,多头,addnorm的输入并集
    def __init__(self, q_size, k_size, v_size, heads, num_hiddens, dropout, inputs, hiddens, normalized_shape, expansion, **kwargs):
        super(EncoderBlock_vit, self).__init__(**kwargs)
        self.attention = MultiHeadsAtten_vit(q_size, k_size, v_size, heads, num_hiddens, dropout)
        self.num_hiddens = normalized_shape
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        # 这里的outputs 是多头的输出维度
        self.fnn = FFN(inputs, hiddens, num_hiddens, expansion)
        self.addnorm2 = AddNorm(num_hiddens, dropout)


    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, valid_lens))
        return self.addnorm2(Y, self.fnn(Y))


# batch_size = 2
# nums = 196
# num_hiddens = 768

# # 使用 torch.randn 生成具有随机值的张量
# q = torch.randn(batch_size, nums, num_hiddens)
# k = torch.randn(batch_size, nums, num_hiddens)
# v = torch.randn(batch_size, nums, num_hiddens)
# patches_embedded = PatchesEmbedding(224, 3, 16, 768)(torch.ones([1, 3, 224, 224]))
# print("patches_embedding's shape: ", patches_embedded.shape)
# print(MultiHeadsAtten_vit(768, 768, 768, 8, 768)(patches_embedded).shape)
# # encoder_blk = EncoderBlock_vit(768, 768, 768, 8, 768, 0.1, 768, 1024, 768, 2)
# print(EncoderBlock_vit(768, 768, 768, 8, 768, 0.1, 768, 1024, 768, 2)(patches_embedded, None).shape)
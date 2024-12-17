import torch
import torch.nn as nn

# 每一个block里面包含ffn和多头注意力，他们之间使用addnorm和残差连接
class EncoderBlock(nn.Module):
    # 所以输入是ffn,多头,addnorm的输入并集
    def __init__(self, q, k, v, heads, num_hiddens, dropout, inputs, hiddens, normalized_shape, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(q, k, v, heads, num_hiddens, dropout)
        self.addnorm1 = AddNorm(normalized_shape, dropout)
        # 这里的outputs 是多头的输出维度
        self.fnn = FFN(inputs, hiddens, num_hiddens)
        self.addnorm2 = AddNorm(normalized_shape, dropout)


    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.fnn(Y))

# X = torch.ones((2, 100, 24))
# encoder_blk = EncoderBlock(24, 24, 24, 8, 24, 0.5, 24, 48, [100, 24])

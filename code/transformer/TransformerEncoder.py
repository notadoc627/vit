import torch
import torch.nn as nn
import math

class TransformerEncoder(EncoderBlock):
    def __init__(self, vocab_size, q, k, v, heads, num_hiddens, dropout, inputs, hiddens, normalized_shape, num_blocks, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embeddings = nn.Embedding(vocab_size, num_hiddens)
        self.position_encode = positionEncoder(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blocks):
            self.blks.add_module("block"+str(i),
                                EncoderBlock(q, k, v, heads, num_hiddens, dropout, inputs, hiddens, normalized_shape))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.position_encode(self.embeddings(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        # 遍历一个list
        for i, blk in enumerate(self.blks):
            # blk是一个EncoderBlock，它的forward里面就是只传入X
            X = blk(X, valid_lens)
            # 调用encoder里面的attention，再调用多头里面的，再调用dot里面
            self.attention_weights[i] = blk.attention.attention.attention_weights

        return X


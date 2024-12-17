import torch
import torch.nn as nn
from code.transformer.EncoderBlock_vit import EncoderBlock_vit
from code.PatchesEmbedding import PatchesEmbedding

class TransformerEncoder(nn.Module):
    def __init__(self, depth, **kwargs):
        super(TransformerEncoder, self).__init__()
        # 创建一个包含多个EncoderBlock_vit的列表
        encoder_blocks = [EncoderBlock_vit(**kwargs) for _ in range(depth)]
        # 使用nn.Sequential将这些blocks组合起来
        self.encoders = nn.Sequential(*encoder_blocks)

    def forward(self, X, valid_lens=None):
        # 依次通过每个EncoderBlock_vit，确保传递X和valid_lens
        for encoder in self.encoders:
            X = encoder(X, valid_lens)
        return X

# patches_embedded = PatchesEmbedding(224, 3, 16, 768)(torch.ones([1, 3, 224, 224]))
# # 定义参数
# q_size = 768
# k_size = 768
# v_size = 768
# heads = 8
# num_hiddens = 768
# dropout = 0.1
# inputs = 768
# hiddens = 3072  # 这是FFN中的隐藏层维度
# normalized_shape = 768  # 这是AddNorm的输入维度
# expansion = 4  # 这是FFN中的扩张因子
# depth = 12  # 编码器层数
#
# # 实例化TransformerEncoder
# transformer_encoder = TransformerEncoder(
#     depth=depth,
#     q_size=q_size,
#     k_size=k_size,
#     v_size=v_size,
#     heads=heads,
#     num_hiddens=num_hiddens,
#     dropout=dropout,
#     inputs=inputs,
#     hiddens=hiddens,
#     normalized_shape=normalized_shape,
#     expansion=expansion
# )
#
# print(transformer_encoder.forward(patches_embedded, None).shape)
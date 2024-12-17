import torch
import torch.nn as nn
from code.PatchesEmbedding import PatchesEmbedding  # 确保模块存在并正确导入
from code.transformer.TransformerEncoder_vit import TransformerEncoder  # 确保模块存在并正确导入
from code.transformer.ClassificationHead import ClassificationHead  # 确保模块存在并正确导入

class ViT(nn.Module):
    def __init__(self,
                in_channels,
                patch_size,
                num_hiddens,
                img_size,
                depth,
                num_classes,
                q_size,
                k_size,
                v_size,
                heads,
                dropout,
                inputs,
                hiddens,
                normalized_shape,
                expansion):
        super(ViT, self).__init__()
        # 定义各个子模块
        self.patch_embedding = PatchesEmbedding(img_size, in_channels, patch_size, num_hiddens)  # 使用 PatchesEmbedding
        self.transformer_encoder = TransformerEncoder(
                depth=depth,
                q_size=q_size,
                k_size=k_size,
                v_size=v_size,
                heads=heads,
                num_hiddens=num_hiddens,
                dropout=dropout,
                inputs=inputs,
                hiddens=hiddens,
                normalized_shape=normalized_shape,
                expansion=expansion
             )
        self.classification_head = ClassificationHead(num_hiddens, num_classes)

    def forward(self, X):
        # 按顺序应用各个子模块
        X = self.patch_embedding(X)
        X = self.transformer_encoder(X, None)
        X = self.classification_head(X)
        return X

# # 测试 PatchesEmbedding
# patches_embedded = PatchesEmbedding(224, 3, 16, 768)(torch.ones([1, 3, 224, 224]))
#
# # 定义参数
# in_channels = 3
# patch_size = 16
# q_size = 768
# k_size = 768
# v_size = 768
# heads = 8
# num_hiddens = 768
# dropout = 0.1
# inputs = 768
# hiddens = 3072  # 这是 FFN 中的隐藏层维度
# normalized_shape = 768  # 这是 AddNorm 的输入维度
# expansion = 4  # 这是 FFN 中的扩张因子
# depth = 12  # 编码器层数
# num_classes = 2
# img_size = 224
#
# # 实例化 TransformerEncoder
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
# # 检查 transformer_encoder 的输出形状
# print(transformer_encoder.forward(patches_embedded, None).shape)
#
# # 实例化 ViT
# vit = ViT(
#                 in_channels=in_channels,
#                 patch_size=patch_size,
#                 num_hiddens=num_hiddens,
#                 img_size=img_size,
#                 depth=depth,
#                 num_classes=num_classes,
#                 q_size=q_size,
#                 k_size=k_size,
#                 v_size=v_size,
#                 heads=heads,
#                 dropout=dropout,
#                 inputs=inputs,
#                 hiddens=hiddens,
#                 normalized_shape=normalized_shape,
#                 expansion=expansion
# )
#
# # 使用 ViT 模型进行前向传播
# output = vit(torch.ones([1, 3, 224, 224]))
# print(output.shape)  # 检查输出形状是否正确

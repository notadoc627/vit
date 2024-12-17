import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, inputs, hiddens, outputs, expansion, **kwargs):
        super(FFN, self).__init__(**kwargs)
        self.fcn1 = nn.Linear(inputs, expansion * hiddens)
        self.relu = nn.ReLU()
        self.fcn2 = nn.Linear(expansion * hiddens, outputs)

    def forward(self, X):
        # X input shape (batchsize, lens, features)
        # X output shape (batchsize, lens, outputs)
        return self.fcn2(self.relu(self.fcn1(X)))

# ffn = FNN(4,4,8)
# ffn.eval()
# # 获取batch = 1的输出
# print(ffn(torch.ones((2, 3, 4)))[0])
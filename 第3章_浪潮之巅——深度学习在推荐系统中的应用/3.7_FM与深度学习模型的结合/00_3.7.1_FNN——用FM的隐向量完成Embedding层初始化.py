# 00_3.7.1 FNN——用FM的隐向量完成Embedding层初始化

"""
Lecture: 第3章 浪潮之巅——深度学习在推荐系统中的应用/3.7 FM与深度学习模型的结合
Content: 00_3.7.1 FNN——用FM的隐向量完成Embedding层初始化
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# 定义FM模型
class FM(nn.Module):
    def __init__(self, num_features, k):
        super(FM, self).__init__()
        self.num_features = num_features
        self.k = k
        self.linear = nn.Linear(num_features, 1)
        self.v = nn.Parameter(torch.randn(num_features, k))

    def forward(self, x):
        linear_part = self.linear(x)
        interactions_part_1 = torch.pow(torch.matmul(x, self.v), 2)
        interactions_part_2 = torch.matmul(torch.pow(x, 2), torch.pow(self.v, 2))
        interactions_part = 0.5 * torch.sum(interactions_part_1 - interactions_part_2, dim=1, keepdim=True)
        output = linear_part + interactions_part
        return output

# 定义FNN模型
class FNN(nn.Module):
    def __init__(self, num_features, k, hidden_dims, fm_model):
        super(FNN, self).__init__()
        self.num_features = num_features
        self.k = k

        # 使用FM模型的隐向量初始化Embedding层
        self.embeddings = nn.Parameter(fm_model.v.clone().detach())
        
        # 定义全连接层
        layers = []
        input_dim = num_features * k
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x_embed = torch.matmul(x, self.embeddings)
        x_embed = x_embed.view(x.size(0), -1)
        output = self.layers(x_embed)
        return output

# 数据准备
num_features = 100000
k = 32
hidden_dims = [64, 32]
batch_size = 64
num_epochs = 10

# 生成示例数据
X = torch.randn(batch_size, num_features)
y = torch.randn(batch_size, 1)

# 训练FM模型
fm_model = FM(num_features, k)
optimizer_fm = optim.Adam(fm_model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    fm_model.train()
    optimizer_fm.zero_grad()
    outputs = fm_model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer_fm.step()
    print(f"FM Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 训练FNN模型
fnn_model = FNN(num_features, k, hidden_dims, fm_model)
optimizer_fnn = optim.Adam(fnn_model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    fnn_model.train()
    optimizer_fnn.zero_grad()
    outputs = fnn_model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer_fnn.step()
    print(f"FNN Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
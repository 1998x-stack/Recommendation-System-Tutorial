# 01_3.7.2 DeepFM——用FM代替Wide部分

"""
Lecture: 第3章 浪潮之巅——深度学习在推荐系统中的应用/3.7 FM与深度学习模型的结合
Content: 01_3.7.2 DeepFM——用FM代替Wide部分
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

class DeepFM(nn.Module):
    def __init__(self, num_features, k, hidden_dims):
        super(DeepFM, self).__init__()
        self.num_features = num_features
        self.k = k

        # FM部分
        self.fm_linear = nn.Linear(num_features, 1)
        self.fm_v = nn.Parameter(torch.randn(num_features, k))

        # DNN部分
        self.embeddings = nn.Embedding(num_features, k)
        self.dnn_input_dim = num_features * k
        layers = []
        input_dim = self.dnn_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        # FM部分
        fm_linear_part = self.fm_linear(x)
        fm_interactions_part_1 = torch.pow(torch.matmul(x, self.fm_v), 2)
        fm_interactions_part_2 = torch.matmul(torch.pow(x, 2), torch.pow(self.fm_v, 2))
        fm_interactions_part = 0.5 * torch.sum(fm_interactions_part_1 - fm_interactions_part_2, dim=1, keepdim=True)
        fm_output = fm_linear_part + fm_interactions_part

        # DNN部分
        x_embed = self.embeddings(x.long()).view(-1, self.dnn_input_dim)
        dnn_output = self.dnn(x_embed)

        # 输出
        output = fm_output + dnn_output
        return output

# 数据准备
num_features = 100000
k = 32
hidden_dims = [64, 32]
batch_size = 64
num_epochs = 10

# 生成示例数据
X = torch.randint(0, num_features, (batch_size, num_features)).float()
y = torch.randn(batch_size, 1)

# 初始化模型
model = DeepFM(num_features, k, hidden_dims)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练模型
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
# 02_3.7.3 NFM——FM的神经网络化尝试

"""
Lecture: 第3章 浪潮之巅——深度学习在推荐系统中的应用/3.7 FM与深度学习模型的结合
Content: 02_3.7.3 NFM——FM的神经网络化尝试
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BiInteractionPooling(nn.Module):
    def __init__(self):
        super(BiInteractionPooling, self).__init__()

    def forward(self, x):
        square_of_sum = torch.pow(torch.sum(x, dim=1), 2)
        sum_of_square = torch.sum(torch.pow(x, 2), dim=1)
        bi_interaction = 0.5 * (square_of_sum - sum_of_square)
        return bi_interaction

class NFM(nn.Module):
    def __init__(self, num_features, k, hidden_dims):
        super(NFM, self).__init__()
        self.num_features = num_features
        self.k = k

        # Embedding层
        self.embeddings = nn.Embedding(num_features, k)

        # 特征交叉池化层
        self.bi_interaction_pooling = BiInteractionPooling()

        # 全连接层
        layers = []
        input_dim = k
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        x_embed = self.embeddings(x).view(-1, self.num_features, self.k)
        bi_interaction = self.bi_interaction_pooling(x_embed)
        output = self.dnn(bi_interaction)
        return output

# 数据准备
num_features = 10000
k = 32
hidden_dims = [64, 32]
batch_size = 64
num_epochs = 10

# 生成示例数据
X = torch.randint(0, num_features, (batch_size, num_features))
y = torch.randn(batch_size, 1)

# 初始化模型
model = NFM(num_features, k, hidden_dims)
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

# 00_3.8.1 AFM——引入注意力机制的FM

"""
Lecture: 第3章 浪潮之巅——深度学习在推荐系统中的应用/3.8 注意力机制在推荐模型中的应用
Content: 00_3.8.1 AFM——引入注意力机制的FM
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

class AttentionNet(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionNet, self).__init__()
        self.attention_fc = nn.Linear(embed_dim, 1)
    
    def forward(self, x):
        attention_score = self.attention_fc(x)
        attention_weight = F.softmax(attention_score, dim=1)
        return attention_weight

class AFM(nn.Module):
    def __init__(self, num_features, embed_dim):
        super(AFM, self).__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim

        # Embedding层
        self.embeddings = nn.Embedding(num_features, embed_dim)

        # 特征交叉池化层
        self.bi_interaction_pooling = BiInteractionPooling()

        # 注意力网络
        self.attention_net = AttentionNet(embed_dim)

        # 输出层
        self.output_fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x_embed = self.embeddings(x).view(-1, self.num_features, self.embed_dim)
        bi_interaction = self.bi_interaction_pooling(x_embed)
        attention_weight = self.attention_net(bi_interaction)
        weighted_pooling = torch.sum(attention_weight * bi_interaction, dim=1)
        output = self.output_fc(weighted_pooling)
        return output

# 数据准备
num_features = 10000
embed_dim = 32
batch_size = 64
num_epochs = 10

# 生成示例数据
X = torch.randint(0, num_features, (batch_size, num_features))
y = torch.randn(batch_size, 1)

# 初始化模型
model = AFM(num_features, embed_dim)
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
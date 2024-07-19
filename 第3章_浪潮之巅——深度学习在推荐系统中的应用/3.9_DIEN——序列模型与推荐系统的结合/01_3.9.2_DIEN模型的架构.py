# 01_3.9.2 DIEN模型的架构

"""
Lecture: 第3章 浪潮之巅——深度学习在推荐系统中的应用/3.9 DIEN——序列模型与推荐系统的结合
Content: 01_3.9.2 DIEN模型的架构
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class InterestExtractorLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        初始化兴趣抽取层
        :param input_dim: 输入的维度大小
        :param hidden_dim: 隐藏层的维度大小
        """
        super(InterestExtractorLayer, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        前向传播函数
        :param x: 输入的行为序列
        :return: GRU的输出和隐藏状态
        """
        output, hidden = self.gru(x)
        return output, hidden

class AttentionNet(nn.Module):
    def __init__(self, hidden_dim: int):
        """
        初始化注意力网络
        :param hidden_dim: 隐藏层的维度大小
        """
        super(AttentionNet, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        :param hidden: 用户兴趣状态向量
        :param target: 目标广告的向量
        :return: 注意力权重
        """
        # 计算注意力得分
        scores = self.fc(hidden).squeeze(-1)
        scores = torch.bmm(target.unsqueeze(1), scores.unsqueeze(2)).squeeze(2)
        # 计算注意力权重
        weights = F.softmax(scores, dim=1)
        return weights

class AUGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        初始化AUGRU单元
        :param input_dim: 输入的维度大小
        :param hidden_dim: 隐藏层的维度大小
        """
        super(AUGRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, attn_weight: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        :param x: 当前时间步的输入
        :param hidden: 上一时间步的隐藏状态
        :param attn_weight: 当前时间步的注意力权重
        :return: 更新后的隐藏状态
        """
        # 计算重置门
        reset_gate = torch.sigmoid(self.fc(hidden))
        # 更新隐藏状态
        updated_hidden = (1 - reset_gate) * hidden + reset_gate * self.gru_cell(x, hidden)
        # 加权更新隐藏状态
        updated_hidden = attn_weight * updated_hidden + (1 - attn_weight) * hidden
        return updated_hidden

class InterestEvolvingLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        初始化兴趣进化层
        :param input_dim: 输入的维度大小
        :param hidden_dim: 隐藏层的维度大小
        """
        super(InterestEvolvingLayer, self).__init__()
        self.augru = AUGRUCell(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, attn_weight: torch.Tensor) -> tuple:
        """
        前向传播函数
        :param x: 输入的行为序列
        :param hidden: 初始隐藏状态
        :param attn_weight: 注意力权重
        :return: 序列输出和最后一个时间步的隐藏状态
        """
        seq_len = x.size(1)
        outputs = []
        for t in range(seq_len):
            hidden = self.augru(x[:, t, :], hidden, attn_weight[:, t])
            outputs.append(hidden.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs, hidden

class DIEN(nn.Module):
    def __init__(self, num_features: int, embed_dim: int, hidden_dim: int):
        """
        初始化DIEN模型
        :param num_features: 特征数量
        :param embed_dim: Embedding层的维度大小
        :param hidden_dim: 隐藏层的维度大小
        """
        super(DIEN, self).__init__()
        self.embeddings = nn.Embedding(num_features, embed_dim)
        self.interest_extractor = InterestExtractorLayer(embed_dim, hidden_dim)
        self.attention_net = AttentionNet(hidden_dim)
        self.interest_evolving = InterestEvolvingLayer(embed_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_hist: torch.Tensor, ad_feature: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        :param user_hist: 用户行为序列
        :param ad_feature: 目标广告特征
        :return: 预测输出
        """
        user_embed = self.embeddings(user_hist)
        ad_embed = self.embeddings(ad_feature).unsqueeze(1)

        interest_output, interest_hidden = self.interest_extractor(user_embed)
        attn_weight = self.attention_net(interest_output, ad_embed)

        evolving_output, evolving_hidden = self.interest_evolving(user_embed, interest_hidden[-1], attn_weight)
        output = self.fc(evolving_hidden)
        return output

# 数据准备
num_features = 10000
embed_dim = 32
hidden_dim = 64
batch_size = 64
num_epochs = 10
seq_length = 10

# 生成示例数据
user_hist = torch.randint(0, num_features, (batch_size, seq_length))
ad_feature = torch.randint(0, num_features, (batch_size,))
y = torch.randn(batch_size, 1)

# 初始化模型
model = DIEN(num_features, embed_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练模型
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(user_hist, ad_feature)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


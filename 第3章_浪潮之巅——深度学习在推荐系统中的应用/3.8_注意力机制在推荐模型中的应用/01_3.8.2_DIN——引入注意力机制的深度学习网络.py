# 01_3.8.2 DIN——引入注意力机制的深度学习网络

"""
Lecture: 第3章 浪潮之巅——深度学习在推荐系统中的应用/3.8 注意力机制在推荐模型中的应用
Content: 01_3.8.2 DIN——引入注意力机制的深度学习网络
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class AttentionNet(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionNet, self).__init__()
        self.attention_fc = nn.Linear(embed_dim * 2, 1)
    
    def forward(self, user_embed, ad_embed):
        concat = torch.cat((user_embed, ad_embed), dim=-1)
        attention_score = self.attention_fc(concat)
        attention_weight = F.softmax(attention_score, dim=1)
        return attention_weight

class DIN(nn.Module):
    def __init__(self, num_features, embed_dim):
        super(DIN, self).__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim

        # Embedding层
        self.embeddings = nn.Embedding(num_features, embed_dim)

        # 注意力网络
        self.attention_net = AttentionNet(embed_dim)

        # 输出层
        self.output_fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_hist, ad_feature):
        user_embed = self.embeddings(user_hist).view(-1, user_hist.size(1), self.embed_dim)
        ad_embed = self.embeddings(ad_feature).view(-1, self.embed_dim)

        attention_weight = self.attention_net(user_embed, ad_embed)
        weighted_user_embed = torch.sum(attention_weight * user_embed, dim=1)

        concat = torch.cat((weighted_user_embed, ad_embed), dim=-1)
        output = self.output_fc(concat)
        return output

# 数据准备
num_features = 10000
embed_dim = 32
batch_size = 64
num_epochs = 10
seq_length = 10

# 生成示例数据
user_hist = torch.randint(0, num_features, (batch_size, seq_length))
ad_feature = torch.randint(0, num_features, (batch_size,))
y = torch.randn(batch_size, 1)

# 初始化模型
model = DIN(num_features, embed_dim)
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
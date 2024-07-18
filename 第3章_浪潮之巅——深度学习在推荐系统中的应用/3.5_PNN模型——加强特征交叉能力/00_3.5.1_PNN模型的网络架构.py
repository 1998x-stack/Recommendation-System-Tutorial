# 00_3.5.1 PNN模型的网络架构

"""
Lecture: 第3章 浪潮之巅——深度学习在推荐系统中的应用/3.5 PNN模型——加强特征交叉能力
Content: 00_3.5.1 PNN模型的网络架构
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Tuple, List

class CustomDataset(Dataset):
    """自定义数据集类，用于加载用户、物品和评分数据。
    
    Args:
        users (np.ndarray): 用户ID向量。
        items (np.ndarray): 物品ID向量。
        ratings (np.ndarray): 评分向量。
    """
    
    def __init__(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self) -> int:
        return len(self.users)
    
    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
        return self.users[idx], self.items[idx], self.ratings[idx]

class PNN(nn.Module):
    """PNN模型定义，包含内积和外积操作的Product层。
    
    Args:
        num_users (int): 用户数量。
        num_items (int): 物品数量。
        embedding_dim (int): Embedding向量维度。
        hidden_layers (List[int]): 隐层神经元数量。
    """
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, hidden_layers: List[int]):
        super(PNN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.embedding_dim = embedding_dim
        
        # 全连接层定义
        input_dim = 2 * embedding_dim + embedding_dim * (embedding_dim - 1) // 2
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.fc_layers = nn.Sequential(*layers)
        
    def forward(self, user: torch.LongTensor, item: torch.LongTensor) -> torch.FloatTensor:
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        
        # 内积操作
        inner_product = torch.sum(user_emb * item_emb, dim=1, keepdim=True)
        
        # 外积操作并降维
        outer_product = torch.bmm(user_emb.unsqueeze(2), item_emb.unsqueeze(1)) # .view(-1, self.embedding_dim ** 2)
        outer_product = outer_product[:, torch.tril(torch.ones(self.embedding_dim, self.embedding_dim)).bool()].view(-1, self.embedding_dim * (self.embedding_dim - 1) // 2)
        
        # 拼接操作
        product_layer = torch.cat([user_emb, item_emb, inner_product, outer_product], dim=1)
        
        # 全连接层
        output = self.fc_layers(product_layer)
        return torch.sigmoid(output)

def train_model(model: nn.Module, dataloader: DataLoader, criterion: nn.BCELoss, optimizer: optim.Adam, epochs: int) -> None:
    """训练PNN模型。
    
    Args:
        model (nn.Module): PNN模型。
        dataloader (DataLoader): 训练数据的DataLoader。
        criterion (nn.BCELoss): 损失函数。
        optimizer (optim.Adam): 优化器。
        epochs (int): 训练轮数。
    """
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for user, item, rating in dataloader:
            optimizer.zero_grad()
            outputs = model(user, item)
            loss = criterion(outputs, rating.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}')

def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.BCELoss) -> None:
    """评估PNN模型。
    
    Args:
        model (nn.Module): PNN模型。
        dataloader (DataLoader): 验证数据的DataLoader。
        criterion (nn.BCELoss): 损失函数。
    """
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for user, item, rating in dataloader:
            outputs = model(user, item)
            loss = criterion(outputs, rating.unsqueeze(1))
            eval_loss += loss.item()
    print(f'Evaluation Loss: {eval_loss / len(dataloader):.4f}')

def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载和预处理数据。
    
    Args:
        file_path (str): 数据文件路径。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 训练集和验证集的用户、物品和评分数据。
    """
    data = pd.read_csv(file_path)
    users = data['user_id'].values
    items = data['item_id'].values
    ratings = data['rating'].values
    split_idx = int(len(data) * 0.8)
    return users[:split_idx], items[:split_idx], ratings[:split_idx], users[split_idx:], items[split_idx:], ratings[split_idx:]

def main() -> None:
    """主函数，执行PNN模型的训练和评估。"""
    # 加载数据
    users_train, items_train, ratings_train, users_val, items_val, ratings_val = load_data('data.csv')
    
    # 创建Dataset和DataLoader
    train_dataset = CustomDataset(users_train, items_train, ratings_train)
    val_dataset = CustomDataset(users_val, items_val, ratings_val)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 模型初始化
    num_users = max(users_train.max(), users_val.max()) + 1
    num_items = max(items_train.max(), items_val.max()) + 1
    model = PNN(num_users, num_items, embedding_dim=32, hidden_layers=[64, 32, 16])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_dataloader, criterion, optimizer, epochs=20)

    # 评估模型
    evaluate_model(model, val_dataloader, criterion)

if __name__ == '__main__':
    main()
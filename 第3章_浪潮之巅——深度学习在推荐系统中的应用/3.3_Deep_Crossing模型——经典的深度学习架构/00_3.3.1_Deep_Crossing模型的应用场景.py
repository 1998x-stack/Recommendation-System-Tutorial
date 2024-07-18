# 00_3.3.1 Deep Crossing模型的应用场景

"""
Lecture: 第3章 浪潮之巅——深度学习在推荐系统中的应用/3.3 Deep Crossing模型——经典的深度学习架构
Content: 00_3.3.1 Deep Crossing模型的应用场景
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Tuple

class CustomDataset(Dataset):
    """自定义数据集类，用于加载特征和标签。
    
    Args:
        features (np.ndarray): 特征矩阵。
        labels (np.ndarray): 标签向量。
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return self.features[idx], self.labels[idx]

class EmbeddingLayer(nn.Module):
    """Embedding层，用于将稀疏类别型特征转换为稠密向量。
    
    Args:
        num_embeddings (int): 类别数量。
        embedding_dim (int): Embedding向量维度。
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
    
    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        return self.embedding(x)

class ResidualUnit(nn.Module):
    """残差单元，用于特征自动交叉组合。
    
    Args:
        input_dim (int): 输入特征维度。
    """
    
    def __init__(self, input_dim: int):
        super(ResidualUnit, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim)
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return self.relu(out + residual)

class DeepCrossing(nn.Module):
    """Deep Crossing模型定义。
    
    Args:
        input_dim (int): 输入特征维度。
        embedding_dims (List[int]): 各类别型特征的Embedding向量维度。
        num_residual_units (int): 残差单元的数量。
    """
    
    def __init__(self, input_dim: int, embedding_dims: list, num_residual_units: int):
        super(DeepCrossing, self).__init__()
        self.embedding_layers = nn.ModuleList([EmbeddingLayer(num_emb, emb_dim) for num_emb, emb_dim in embedding_dims])
        self.input_dim = input_dim + sum([emb_dim for _, emb_dim in embedding_dims])
        self.residual_units = nn.Sequential(*[ResidualUnit(self.input_dim) for _ in range(num_residual_units)])
        self.output_layer = nn.Linear(self.input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_categorical: torch.LongTensor, x_numerical: torch.FloatTensor) -> torch.FloatTensor:
        embeddings = [emb(x_categorical[:, i]) for i, emb in enumerate(self.embedding_layers)]
        embeddings = torch.cat(embeddings, dim=1)
        x = torch.cat([embeddings, x_numerical], dim=1)
        x = self.residual_units(x)
        x = self.output_layer(x)
        return self.sigmoid(x)

def train_model(model: nn.Module, dataloader: DataLoader, criterion: nn.BCELoss, optimizer: optim.Adam, epochs: int) -> None:
    """训练Deep Crossing模型。
    
    Args:
        model (nn.Module): Deep Crossing模型。
        dataloader (DataLoader): 训练数据的DataLoader。
        criterion (nn.BCELoss): 损失函数。
        optimizer (optim.Adam): 优化器。
        epochs (int): 训练轮数。
    """
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x_numerical, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(x_categorical, x_numerical)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}')

def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.BCELoss) -> None:
    """评估Deep Crossing模型。
    
    Args:
        model (nn.Module): Deep Crossing模型。
        dataloader (DataLoader): 验证数据的DataLoader。
        criterion (nn.BCELoss): 损失函数。
    """
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for x_numerical, labels in dataloader:
            outputs = model(x_categorical, x_numerical)
            loss = criterion(outputs, labels.unsqueeze(1))
            eval_loss += loss.item()
    print(f'Evaluation Loss: {eval_loss / len(dataloader):.4f}')

def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载和预处理数据。
    
    Args:
        file_path (str): 数据文件路径。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 特征和标签的训练集和验证集。
    """
    data = pd.read_csv(file_path)
    # 假设数据集已经进行了一些预处理，包含类别型特征和数值型特征的编码
    x_categorical = data[['query', 'keyword', 'title', 'landing_page', 'match_type']].values
    x_numerical = data[['click_rate', 'predicted_click_rate', 'budget', 'impression', 'click']].values
    y = data['label'].values
    split_idx = int(len(data) * 0.8)
    return x_categorical[:split_idx], x_numerical[:split_idx], y[:split_idx], x_categorical[split_idx:], x_numerical[split_idx:], y[split_idx:]

def main() -> None:
    """主函数，执行Deep Crossing模型的训练和评估。"""
    # 加载数据
    x_categorical_train, x_numerical_train, y_train, x_categorical_val, x_numerical_val, y_val = load_data('data.csv')
    
    # 创建Dataset和DataLoader
    train_dataset = CustomDataset(x_categorical_train, x_numerical_train, y_train)
    val_dataset = CustomDataset(x_categorical_val, x_numerical_val, y_val)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 模型初始化
    embedding_dims = [(10000, 32), (10000, 32), (10000, 32), (10000, 32), (10, 5)]  # 类别型特征的Embedding维度
    model = DeepCrossing(input_dim=5, embedding_dims=embedding_dims, num_residual_units=5)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_dataloader, criterion, optimizer, epochs=20)

    # 评估模型
    evaluate_model(model, val_dataloader, criterion)

if __name__ == '__main__':
    main()
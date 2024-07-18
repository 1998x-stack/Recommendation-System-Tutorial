# 02_3.6.3 Wide&Deep模型的进化——Deep&Cross模型

"""
Lecture: 第3章 浪潮之巅——深度学习在推荐系统中的应用/3.6 Wide&Deep 模型——记忆能力和泛化能力的综合
Content: 02_3.6.3 Wide&Deep模型的进化——Deep&Cross模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import List, Tuple

class CustomDataset(Dataset):
    """自定义数据集类，用于加载特征和标签数据。
    
    Args:
        wide_features (np.ndarray): Wide部分的输入特征。
        deep_features (np.ndarray): Deep部分的输入特征。
        labels (np.ndarray): 标签（目标值）。
    """
    
    def __init__(self, wide_features: np.ndarray, deep_features: np.ndarray, labels: np.ndarray):
        self.wide_features = torch.FloatTensor(wide_features)
        self.deep_features = torch.LongTensor(deep_features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        return self.wide_features[idx], self.deep_features[idx], self.labels[idx]

class CrossNetwork(nn.Module):
    """Cross网络定义。
    
    Args:
        input_dim (int): 输入特征的维度。
        num_layers (int): 交叉层的数量。
    """
    
    def __init__(self, input_dim: int, num_layers: int):
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x0 = x
        for i in range(self.num_layers):
            xl_w = self.cross_layers[i](x)
            x = x0 * xl_w + self.bias[i] + x
        return x

class DeepNetwork(nn.Module):
    """Deep网络定义。
    
    Args:
        num_embeddings (int): Embedding层输入的类别数量。
        embedding_dim (int): Embedding层输出的维度。
        hidden_layers (List[int]): 隐藏层的神经元数量。
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, hidden_layers: List[int]):
        super(DeepNetwork, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        input_dim = embedding_dim * num_embeddings
        self.deep_layers = nn.Sequential()
        
        for i, hidden_dim in enumerate(hidden_layers):
            self.deep_layers.add_module(f'fc{i}', nn.Linear(input_dim, hidden_dim))
            self.deep_layers.add_module(f'relu{i}', nn.ReLU())
            input_dim = hidden_dim
            
        self.deep_layers.add_module('output', nn.Linear(input_dim, 1))
        
    def forward(self, deep_input: torch.LongTensor) -> torch.FloatTensor:
        deep_input_emb = self.embedding(deep_input).view(deep_input.size(0), -1)
        deep_output = self.deep_layers(deep_input_emb)
        return deep_output

class DeepAndCrossModel(nn.Module):
    """Deep & Cross模型定义。
    
    Args:
        input_dim_wide (int): Wide部分的输入特征维度。
        num_embeddings (int): Deep部分Embedding层输入的类别数量。
        embedding_dim (int): Deep部分Embedding层输出的维度。
        cross_layers (int): Cross部分交叉层的数量。
        hidden_layers (List[int]): Deep部分隐藏层的神经元数量。
    """
    
    def __init__(self, input_dim_wide: int, num_embeddings: int, embedding_dim: int, cross_layers: int, hidden_layers: List[int]):
        super(DeepAndCrossModel, self).__init__()
        
        # Cross部分
        self.cross_network = CrossNetwork(input_dim_wide, cross_layers)
        
        # Deep部分
        self.deep_network = DeepNetwork(num_embeddings, embedding_dim, hidden_layers)
        
        # 输出层
        self.output_layer = nn.Linear(input_dim_wide + 1, 1)
        
    def forward(self, wide_input: torch.FloatTensor, deep_input: torch.LongTensor) -> torch.FloatTensor:
        # Cross网络
        cross_output = self.cross_network(wide_input)
        
        # Deep网络
        deep_output = self.deep_network(deep_input)
        
        # Wide & Deep结合
        combined_output = torch.cat((cross_output, deep_output), dim=1)
        output = torch.sigmoid(self.output_layer(combined_output))
        return output

def train_model(model: nn.Module, dataloader: DataLoader, criterion: nn.BCELoss, optimizer: optim.Adam, epochs: int) -> None:
    """训练Deep & Cross模型。
    
    Args:
        model (nn.Module): Deep & Cross模型。
        dataloader (DataLoader): 训练数据的DataLoader。
        criterion (nn.BCELoss): 损失函数。
        optimizer (optim.Adam): 优化器。
        epochs (int): 训练轮数。
    """
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for wide_input, deep_input, label in dataloader:
            optimizer.zero_grad()
            outputs = model(wide_input, deep_input)
            loss = criterion(outputs.squeeze(), label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}')

def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.BCELoss) -> None:
    """评估Deep & Cross模型。
    
    Args:
        model (nn.Module): Deep & Cross模型。
        dataloader (DataLoader): 验证数据的DataLoader。
        criterion (nn.BCELoss): 损失函数。
    """
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for wide_input, deep_input, label in dataloader:
            outputs = model(wide_input, deep_input)
            loss = criterion(outputs.squeeze(), label)
            eval_loss += loss.item()
    print(f'Evaluation Loss: {eval_loss / len(dataloader):.4f}')

def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载和预处理数据。
    
    Args:
        file_path (str): 数据文件路径。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 训练集和验证集的特征和标签。
    """
    data = pd.read_csv(file_path)
    wide_features = data.iloc[:, :5].values  # 假设前5列是Wide特征
    deep_features = data.iloc[:, 5:-1].values  # 假设5列之后到倒数第二列是Deep特征
    labels = data.iloc[:, -1].values  # 假设最后一列是标签
    split_idx = int(len(data) * 0.8)
    return (wide_features[:split_idx], deep_features[:split_idx], labels[:split_idx],
            wide_features[split_idx:], deep_features[split_idx:], labels[split_idx:])

def main() -> None:
    """主函数，执行Deep & Cross模型的训练和评估。"""
    # 加载数据
    wide_features_train, deep_features_train, labels_train, wide_features_val, deep_features_val, labels_val = load_data('data.csv')
    
    # 创建Dataset和DataLoader
    train_dataset = CustomDataset(wide_features_train, deep_features_train, labels_train)
    val_dataset = CustomDataset(wide_features_val, deep_features_val, labels_val)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 模型初始化
    num_embeddings = int(deep_features_train.max()) + 1  # 假设Deep特征是类别型特征
    model = DeepAndCrossModel(input_dim_wide=wide_features_train.shape[1], num_embeddings=num_embeddings, embedding_dim=8, cross_layers=3, hidden_layers=[64, 32, 16])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_dataloader, criterion, optimizer, epochs=20)

    # 评估模型
    evaluate_model(model, val_dataloader, criterion)

if __name__ == '__main__':
    main()
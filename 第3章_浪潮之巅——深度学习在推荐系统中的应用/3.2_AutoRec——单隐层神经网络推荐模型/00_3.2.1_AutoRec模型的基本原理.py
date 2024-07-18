# 00_3.2.1 AutoRec模型的基本原理

"""
Lecture: 第3章 浪潮之巅——深度学习在推荐系统中的应用/3.2 AutoRec——单隐层神经网络推荐模型
Content: 00_3.2.1 AutoRec模型的基本原理
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Tuple

def normalize_ratings(ratings: np.ndarray) -> np.ndarray:
    """Normalize the rating matrix to [0, 1] range.

    Args:
        ratings (np.ndarray): Rating matrix.

    Returns:
        np.ndarray: Normalized rating matrix.
    """
    max_rating = np.nanmax(ratings)
    min_rating = np.nanmin(ratings)
    ratings = (ratings - min_rating) / (max_rating - min_rating)
    return ratings

class RatingsDataset(Dataset):
    """Custom Dataset for loading rating matrix."""
    
    def __init__(self, ratings: np.ndarray):
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self) -> int:
        return len(self.ratings)
    
    def __getitem__(self, idx: int) -> torch.FloatTensor:
        return self.ratings[idx]

class AutoRec(nn.Module):
    """AutoRec model definition."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super(AutoRec, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass through the AutoRec model."""
        encoded = self.activation(self.encoder(x))
        decoded = self.activation(self.decoder(encoded))
        return decoded

def train_model(model: nn.Module, dataloader: DataLoader, criterion: nn.MSELoss, optimizer: optim.Adam, epochs: int) -> None:
    """Train the AutoRec model.

    Args:
        model (nn.Module): AutoRec model.
        dataloader (DataLoader): DataLoader for the training data.
        criterion (nn.MSELoss): Loss function.
        optimizer (optim.Adam): Optimizer.
        epochs (int): Number of epochs to train.
    """
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}')

def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.MSELoss) -> None:
    """Evaluate the AutoRec model.

    Args:
        model (nn.Module): AutoRec model.
        dataloader (DataLoader): DataLoader for the evaluation data.
        criterion (nn.MSELoss): Loss function.
    """
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch)
            loss = criterion(outputs, batch)
            eval_loss += loss.item()
    print(f'Evaluation Loss: {eval_loss / len(dataloader):.4f}')

def load_data(file_path: str) -> np.ndarray:
    """Load and preprocess the rating data.

    Args:
        file_path (str): Path to the rating data file.

    Returns:
        np.ndarray: Preprocessed rating matrix.
    """
    ratings = pd.read_csv(file_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    num_users = ratings['user_id'].max()
    num_items = ratings['item_id'].max()
    rating_matrix = np.zeros((num_users, num_items))
    for row in ratings.itertuples():
        rating_matrix[row[1] - 1, row[2] - 1] = row[3]
    rating_matrix = normalize_ratings(rating_matrix)
    return rating_matrix

def main() -> None:
    """Main function to execute the training and evaluation of the AutoRec model."""
    # 加载并预处理示例数据（例如MovieLens 100k数据集）
    rating_matrix = load_data('ml-100k/u.data')

    # 创建Dataset和DataLoader
    dataset = RatingsDataset(rating_matrix)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 模型初始化
    input_dim = rating_matrix.shape[1]
    hidden_dim = 500
    model = AutoRec(input_dim=input_dim, hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, dataloader, criterion, optimizer, epochs=20)

    # 评估模型
    evaluate_model(model, dataloader, criterion)

if __name__ == '__main__':
    main()

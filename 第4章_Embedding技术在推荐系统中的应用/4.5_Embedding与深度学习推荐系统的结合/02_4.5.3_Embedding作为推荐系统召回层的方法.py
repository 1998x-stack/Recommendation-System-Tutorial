# 02_4.5.3 Embedding作为推荐系统召回层的方法

"""
Lecture: 第4章 Embedding技术在推荐系统中的应用/4.5 Embedding与深度学习推荐系统的结合
Content: 02_4.5.3 Embedding作为推荐系统召回层的方法
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple

class YouTubeRecallDataset(Dataset):
    """
    YouTube推荐系统召回层的数据集类，用于存储和提供训练数据。

    Attributes:
        user_histories: 用户观看历史视频的ID列表。
        user_features: 用户的其他特征（如地理位置、年龄、性别等）。
        video_embeddings: 视频的Embedding向量字典。
    """
    def __init__(self, user_histories: List[List[int]], user_features: List[List[float]], video_embeddings: Dict[int, np.ndarray]):
        self.user_histories = user_histories
        self.user_features = user_features
        self.video_embeddings = video_embeddings

    def __len__(self) -> int:
        return len(self.user_histories)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        history = self.user_histories[index]
        features = self.user_features[index]
        video_embs = np.array([self.video_embeddings[vid] for vid in history])
        user_emb = np.mean(video_embs, axis=0)
        user_features = np.array(features)
        return user_emb, user_features

class YouTubeRecallModel(nn.Module):
    """
    YouTube推荐系统召回层的模型类，通过三层ReLU全连接层生成用户Embedding向量。
    
    Attributes:
        input_dim: 输入特征的维度。
        embedding_dim: 嵌入向量的维度。
        hidden_dim: 隐藏层的维度。
    """
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int):
        super(YouTubeRecallModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class YouTubeRecallTrainer:
    """
    YouTube推荐系统召回层的训练器类，负责数据预处理、模型训练和评估。
    
    Attributes:
        user_histories: 用户观看历史视频的ID列表。
        user_features: 用户的其他特征（如地理位置、年龄、性别等）。
        video_embeddings: 视频的Embedding向量字典。
        embedding_dim: 嵌入向量的维度。
        hidden_dim: 隐藏层的维度。
        learning_rate: 学习率。
        epochs: 训练的轮数。
    """
    def __init__(self, user_histories: List[List[int]], user_features: List[List[float]], video_embeddings: Dict[int, np.ndarray], embedding_dim: int, hidden_dim: int, learning_rate: float, epochs: int):
        self.dataset = YouTubeRecallDataset(user_histories, user_features, video_embeddings)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = YouTubeRecallModel(input_dim=embedding_dim + len(user_features[0]), embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self):
        """
        训练YouTube推荐系统召回层的模型。
        """
        data_loader = DataLoader(self.dataset, batch_size=128, shuffle=True)
        
        for epoch in range(self.epochs):
            total_loss = 0
            for user_emb, user_features in data_loader:
                self.optimizer.zero_grad()
                input_data = torch.cat((user_emb, user_features), dim=1)
                output = self.model(input_data)
                loss = self.criterion(output, user_emb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch: {epoch + 1}, Loss: {total_loss:.4f}")

    def get_user_embedding(self, user_emb: np.ndarray, user_features: np.ndarray) -> np.ndarray:
        """
        获取指定用户的嵌入向量。
        
        Args:
            user_emb: 用户观看历史视频的平均Embedding向量。
            user_features: 用户的其他特征。
        
        Returns:
            嵌入向量。
        """
        input_data = torch.tensor(np.concatenate((user_emb, user_features)), dtype=torch.float32)
        embedding_vector = self.model(input_data).detach().numpy()
        return embedding_vector

# 示例数据
user_histories = [
    [1, 2, 3],
    [2, 3, 4],
    [1, 4, 5],
    # 更多用户历史数据...
]

user_features = [
    [0.1, 0.2, 0.3],
    [0.2, 0.3, 0.4],
    [0.1, 0.4, 0.5],
    # 更多用户特征数据...
]

video_embeddings = {
    1: np.random.rand(128),
    2: np.random.rand(128),
    3: np.random.rand(128),
    4: np.random.rand(128),
    5: np.random.rand(128),
    # 更多视频Embedding数据...
}

# 训练YouTube推荐系统召回层的模型
trainer = YouTubeRecallTrainer(user_histories, user_features, video_embeddings, embedding_dim=128, hidden_dim=256, learning_rate=0.001, epochs=10)
trainer.train()

# 获取用户的嵌入向量
user_emb = np.mean([video_embeddings[vid] for vid in user_histories[0]], axis=0)
user_features_example = np.array(user_features[0])
embedding_vector = trainer.get_user_embedding(user_emb, user_features_example)
print(f"User embedding vector: {embedding_vector}")

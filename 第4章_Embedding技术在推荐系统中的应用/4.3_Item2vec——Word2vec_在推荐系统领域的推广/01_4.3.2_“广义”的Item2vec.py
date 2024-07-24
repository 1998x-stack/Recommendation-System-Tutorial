# 01_4.3.2 “广义”的Item2vec

"""
Lecture: 第4章 Embedding技术在推荐系统中的应用/4.3 Item2vec——Word2vec 在推荐系统领域的推广
Content: 01_4.3.2 “广义”的Item2vec
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple

class TwinTowersDataset(Dataset):
    """
    双塔模型数据集类，用于存储和提供训练数据。

    Attributes:
        user_sequences: 用户交互序列列表。
        item_sequences: 物品交互序列列表。
        user_to_idx: 用户到索引的映射字典。
        item_to_idx: 物品到索引的映射字典。
    """
    def __init__(self, user_sequences: List[List[int]], item_sequences: List[List[int]]):
        self.user_sequences = user_sequences
        self.item_sequences = item_sequences
        self.user_to_idx, self.idx_to_user = self._create_vocab(user_sequences)
        self.item_to_idx, self.idx_to_item = self._create_vocab(item_sequences)

    def _create_vocab(self, sequences: List[List[int]]) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        创建物品和索引之间的映射。
        
        Args:
            sequences: 用户或物品的交互序列列表。
        
        Returns:
            token_to_idx: 物品或用户到索引的映射字典。
            idx_to_token: 索引到物品或用户的映射字典。
        """
        token_to_idx = {}
        idx_to_token = {}
        idx = 0
        for seq in sequences:
            for token in seq:
                if token not in token_to_idx:
                    token_to_idx[token] = idx
                    idx_to_token[idx] = token
                    idx += 1
        return token_to_idx, idx_to_token

    def __len__(self) -> int:
        return len(self.user_sequences)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        user_seq = [self.user_to_idx[user] for user in self.user_sequences[index]]
        item_seq = [self.item_to_idx[item] for item in self.item_sequences[index]]
        return user_seq, item_seq

class TwinTowersModel(nn.Module):
    """
    双塔模型类，通过用户塔和物品塔生成嵌入向量。
    
    Attributes:
        user_embedding_dim: 用户嵌入向量的维度。
        item_embedding_dim: 物品嵌入向量的维度。
        user_vocab_size: 用户词汇表大小。
        item_vocab_size: 物品词汇表大小。
        user_embeddings: 用户嵌入层。
        item_embeddings: 物品嵌入层。
    """
    def __init__(self, user_vocab_size: int, item_vocab_size: int, embedding_dim: int):
        super(TwinTowersModel, self).__init__()
        self.user_embedding_dim = embedding_dim
        self.item_embedding_dim = embedding_dim
        self.user_embeddings = nn.Embedding(user_vocab_size, embedding_dim)
        self.item_embeddings = nn.Embedding(item_vocab_size, embedding_dim)

    def forward(self, user_inputs: torch.Tensor, item_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        user_embedding = self.user_embeddings(user_inputs)
        item_embedding = self.item_embeddings(item_inputs)
        return user_embedding, item_embedding

class TwinTowersTrainer:
    """
    双塔模型训练器类，负责数据预处理、模型训练和评估。
    
    Attributes:
        user_sequences: 用户的交互序列。
        item_sequences: 物品的交互序列。
        embedding_dim: 嵌入向量的维度。
        learning_rate: 学习率。
        epochs: 训练的轮数。
    """
    def __init__(self, user_sequences: List[List[int]], item_sequences: List[List[int]], embedding_dim: int, learning_rate: float, epochs: int):
        self.user_sequences = user_sequences
        self.item_sequences = item_sequences
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset = TwinTowersDataset(user_sequences, item_sequences)
        self.user_vocab_size = len(self.dataset.user_to_idx)
        self.item_vocab_size = len(self.dataset.item_to_idx)
        self.model = TwinTowersModel(self.user_vocab_size, self.item_vocab_size, embedding_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def _generate_training_data(self) -> List[Tuple[int, int]]:
        """
        生成训练数据。
        
        Returns:
            training_data: 训练数据对。
        """
        training_data = []
        for user_seq, item_seq in zip(self.user_sequences, self.item_sequences):
            for user, item in zip(user_seq, item_seq):
                user_idx = self.dataset.user_to_idx[user]
                item_idx = self.dataset.item_to_idx[item]
                training_data.append((user_idx, item_idx))
        return training_data

    def train(self):
        """
        训练双塔模型。
        """
        training_data = self._generate_training_data()
        data_loader = DataLoader(training_data, batch_size=128, shuffle=True)
        
        for epoch in range(self.epochs):
            total_loss = 0
            for user_batch, item_batch in data_loader:
                self.optimizer.zero_grad()
                user_batch = user_batch.to(torch.int64)
                item_batch = item_batch.to(torch.int64)
                user_embedding, item_embedding = self.model(user_batch, item_batch)
                loss = self.criterion(user_embedding, item_embedding)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch: {epoch + 1}, Loss: {total_loss:.4f}")

    def get_user_embedding(self, user: int) -> np.ndarray:
        """
        获取指定用户的嵌入向量。
        
        Args:
            user: 用户ID。
        
        Returns:
            嵌入向量。
        """
        user_idx = self.dataset.user_to_idx[user]
        embedding_vector = self.model.user_embeddings.weight[user_idx].detach().numpy()
        return embedding_vector

    def get_item_embedding(self, item: int) -> np.ndarray:
        """
        获取指定物品的嵌入向量。
        
        Args:
            item: 物品ID。
        
        Returns:
            嵌入向量。
        """
        item_idx = self.dataset.item_to_idx[item]
        embedding_vector = self.model.item_embeddings.weight[item_idx].detach().numpy()
        return embedding_vector

# 数据准备
user_sequences = [
    [1, 2, 3, 4, 2],
    [2, 3, 5, 6],
    [1, 4, 2, 5],
    # 更多用户序列...
]

item_sequences = [
    [101, 102, 103, 104, 102],
    [102, 103, 105, 106],
    [101, 104, 102, 105],
    # 更多物品序列...
]

# 训练双塔模型
trainer = TwinTowersTrainer(user_sequences=user_sequences, item_sequences=item_sequences, embedding_dim=50, learning_rate=0.001, epochs=10)
trainer.train()

# 获取用户和物品的嵌入向量
user_id = 1
item_id = 101
user_embedding_vector = trainer.get_user_embedding(user_id)
item_embedding_vector = trainer.get_item_embedding(item_id)
print(f"User {user_id} embedding vector: {user_embedding_vector}")
print(f"Item {item_id} embedding vector: {item_embedding_vector}")

# 00_4.3.1 Item2vec的基本原理

"""
Lecture: 第4章 Embedding技术在推荐系统中的应用/4.3 Item2vec——Word2vec 在推荐系统领域的推广
Content: 00_4.3.1 Item2vec的基本原理
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict

class Item2VecDataset(torch.utils.data.Dataset):
    """
    数据集类，用于存储和提供Item2Vec模型训练所需的数据。
    
    Attributes:
        data: 存储用户的物品交互序列。
        item_to_idx: 物品到索引的映射字典。
        idx_to_item: 索引到物品的映射字典。
    """
    def __init__(self, sequences: List[List[int]]):
        self.data = sequences
        self.item_to_idx, self.idx_to_item = self._create_vocab(sequences)

    def _create_vocab(self, sequences: List[List[int]]) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        创建物品和索引之间的映射。
        
        Args:
            sequences: 用户的物品交互序列列表。
        
        Returns:
            item_to_idx: 物品到索引的映射字典。
            idx_to_item: 索引到物品的映射字典。
        """
        item_to_idx = {}
        idx_to_item = {}
        idx = 0
        for seq in sequences:
            for item in seq:
                if item not in item_to_idx:
                    item_to_idx[item] = idx
                    idx_to_item[idx] = item
                    idx += 1
        return item_to_idx, idx_to_item

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> List[int]:
        return [self.item_to_idx[item] for item in self.data[index]]

class Item2VecModel(nn.Module):
    """
    Item2Vec模型类，通过Skip-gram模型实现物品嵌入。
    
    Attributes:
        embedding_dim: 嵌入向量的维度。
        vocab_size: 词汇表大小。
        embeddings: 嵌入层，用于存储物品的嵌入向量。
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(Item2VecModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_items: torch.Tensor) -> torch.Tensor:
        return self.embeddings(input_items)

class Item2VecTrainer:
    """
    Item2Vec训练器类，负责数据预处理、模型训练和评估。
    
    Attributes:
        sequences: 用户的物品交互序列。
        embedding_dim: 嵌入向量的维度。
        window_size: Skip-gram模型的窗口大小。
        learning_rate: 学习率。
        epochs: 训练的轮数。
    """
    def __init__(self, sequences: List[List[int]], embedding_dim: int, window_size: int, learning_rate: float, epochs: int):
        self.sequences = sequences
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset = Item2VecDataset(sequences)
        self.vocab_size = len(self.dataset.item_to_idx)
        self.model = Item2VecModel(self.vocab_size, embedding_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def _generate_training_data(self) -> List[Tuple[int, int]]:
        """
        生成Skip-gram模型的训练数据。
        
        Returns:
            training_data: Skip-gram模型的训练数据对。
        """
        training_data = []
        for seq in self.sequences:
            for i in range(len(seq)):
                target = self.dataset.item_to_idx[seq[i]]
                context_items = seq[max(0, i - self.window_size): i] + seq[i + 1: min(len(seq), i + 1 + self.window_size)]
                context_items = [self.dataset.item_to_idx[item] for item in context_items]
                for context in context_items:
                    training_data.append((target, context))
        return training_data

    def train(self):
        """
        训练Item2Vec模型。
        """
        training_data = self._generate_training_data()
        data_loader = torch.utils.data.DataLoader(training_data, batch_size=128, shuffle=True)
        
        for epoch in range(self.epochs):
            total_loss = 0
            for target, context in data_loader:
                self.optimizer.zero_grad()
                target = target.to(torch.int64)
                context = context.to(torch.int64)
                output = self.model(target)
                loss = self.criterion(output, context)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch: {epoch + 1}, Loss: {total_loss:.4f}")

    def get_embedding(self, item: int) -> np.ndarray:
        """
        获取指定物品的嵌入向量。
        
        Args:
            item: 物品ID。
        
        Returns:
            嵌入向量。
        """
        item_idx = self.dataset.item_to_idx[item]
        embedding_vector = self.model.embeddings.weight[item_idx].detach().numpy()
        return embedding_vector

# 数据准备
user_sequences = [
    [1, 2, 3, 4, 2],
    [2, 3, 5, 6],
    [1, 4, 2, 5],
    # 更多用户序列...
]

# 训练Item2Vec模型
trainer = Item2VecTrainer(sequences=user_sequences, embedding_dim=50, window_size=2, learning_rate=0.001, epochs=10)
trainer.train()

# 获取物品的嵌入向量
item_id = 1
embedding_vector = trainer.get_embedding(item_id)
print(f"Item {item_id} embedding vector: {embedding_vector}")

# 00_4.4.1 DeepWalk——基础的Graph Embedding方法

"""
Lecture: 第4章 Embedding技术在推荐系统中的应用/4.4 Graph Embedding——引入更多结构信息的图嵌入技术
Content: 00_4.4.1 DeepWalk——基础的Graph Embedding方法
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import networkx as nx
import random

class DeepWalkDataset(Dataset):
    """
    DeepWalk数据集类，用于存储和提供训练数据。

    Attributes:
        walks: 随机游走生成的节点序列。
        word2vec_window: Word2Vec窗口大小。
    """
    def __init__(self, walks: List[List[int]], word2vec_window: int):
        self.walks = walks
        self.word2vec_window = word2vec_window
        self.pairs = self._generate_pairs()

    def _generate_pairs(self) -> List[Tuple[int, int]]:
        """
        根据随机游走生成的节点序列，创建训练对。
        
        Returns:
            pairs: 训练对列表。
        """
        pairs = []
        for walk in self.walks:
            for i, node in enumerate(walk):
                for j in range(1, self.word2vec_window + 1):
                    if i - j >= 0:
                        pairs.append((node, walk[i - j]))
                    if i + j < len(walk):
                        pairs.append((node, walk[i + j]))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[int, int]:
        return self.pairs[index]

class DeepWalkModel(nn.Module):
    """
    DeepWalk模型类，通过Skip-gram模型实现节点嵌入。
    
    Attributes:
        embedding_dim: 嵌入向量的维度。
        vocab_size: 词汇表大小（节点数）。
        embeddings: 嵌入层，用于存储节点的嵌入向量。
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(DeepWalkModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_nodes: torch.Tensor) -> torch.Tensor:
        return self.embeddings(input_nodes)

class DeepWalkTrainer:
    """
    DeepWalk训练器类，负责数据预处理、模型训练和评估。
    
    Attributes:
        graph: 输入的图结构。
        embedding_dim: 嵌入向量的维度。
        walk_length: 每次随机游走的长度。
        num_walks: 每个节点的随机游走次数。
        window_size: Skip-gram模型的窗口大小。
        learning_rate: 学习率。
        epochs: 训练的轮数。
    """
    def __init__(self, graph: nx.Graph, embedding_dim: int, walk_length: int, num_walks: int, window_size: int, learning_rate: float, epochs: int):
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.vocab_size = len(graph.nodes)
        self.model = DeepWalkModel(self.vocab_size, embedding_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.walks = self._generate_walks()
        self.dataset = DeepWalkDataset(self.walks, window_size)

    def _generate_walks(self) -> List[List[int]]:
        """
        在图上进行随机游走，生成节点序列。
        
        Returns:
            walks: 节点序列列表。
        """
        walks = []
        nodes = list(self.graph.nodes)
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self._random_walk(node)
                walks.append(walk)
        return walks

    def _random_walk(self, start_node: int) -> List[int]:
        """
        从指定节点开始进行随机游走。
        
        Args:
            start_node: 起始节点。
        
        Returns:
            walk: 随机游走生成的节点序列。
        """
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            neighbors = list(self.graph.neighbors(cur))
            if len(neighbors) > 0:
                next_node = random.choice(neighbors)
                walk.append(next_node)
            else:
                break
        return walk

    def train(self):
        """
        训练DeepWalk模型。
        """
        data_loader = DataLoader(self.dataset, batch_size=128, shuffle=True)
        
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

    def get_embedding(self, node: int) -> np.ndarray:
        """
        获取指定节点的嵌入向量。
        
        Args:
            node: 节点ID。
        
        Returns:
            嵌入向量。
        """
        node_idx = node
        embedding_vector = self.model.embeddings.weight[node_idx].detach().numpy()
        return embedding_vector

# 数据准备
graph = nx.karate_club_graph()

# 训练DeepWalk模型
trainer = DeepWalkTrainer(graph=graph, embedding_dim=128, walk_length=10, num_walks=80, window_size=5, learning_rate=0.01, epochs=10)
trainer.train()

# 获取节点的嵌入向量
node_id = 0
embedding_vector = trainer.get_embedding(node_id)
print(f"Node {node_id} embedding vector: {embedding_vector}")

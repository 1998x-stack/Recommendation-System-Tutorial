# 02_4.4.3 EGES——阿里巴巴的综合性Graph Embedding方法

"""
Lecture: 第4章 Embedding技术在推荐系统中的应用/4.4 Graph Embedding——引入更多结构信息的图嵌入技术
Content: 02_4.4.3 EGES——阿里巴巴的综合性Graph Embedding方法
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import networkx as nx
import random

class EGESDataset(Dataset):
    """
    EGES数据集类，用于存储和提供训练数据。

    Attributes:
        walks: 随机游走生成的节点序列。
        side_info: 补充信息字典。
        word2vec_window: Word2Vec窗口大小。
    """
    def __init__(self, walks: List[List[int]], side_info: Dict[int, np.ndarray], word2vec_window: int):
        self.walks = walks
        self.side_info = side_info
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

    def __getitem__(self, index: int) -> Tuple[int, int, np.ndarray]:
        target, context = self.pairs[index]
        side_info_vector = self.side_info[target]
        return target, context, side_info_vector

class EGESModel(nn.Module):
    """
    EGES模型类，通过融合图结构信息和补充信息实现节点嵌入。
    
    Attributes:
        embedding_dim: 嵌入向量的维度。
        side_info_dim: 补充信息的维度。
        vocab_size: 词汇表大小（节点数）。
        embeddings: 嵌入层，用于存储节点的嵌入向量。
        side_info_embeddings: 补充信息嵌入层。
        linear: 用于融合图结构嵌入和补充信息嵌入的线性层。
    """
    def __init__(self, vocab_size: int, embedding_dim: int, side_info_dim: int):
        super(EGESModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.side_info_embeddings = nn.Linear(side_info_dim, embedding_dim)
        self.linear = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, target: torch.Tensor, side_info: torch.Tensor) -> torch.Tensor:
        target_embedding = self.embeddings(target)
        side_info_embedding = self.side_info_embeddings(side_info)
        combined_embedding = torch.cat((target_embedding, side_info_embedding), dim=1)
        output = self.linear(combined_embedding)
        return output

class EGESTrainer:
    """
    EGES训练器类，负责数据预处理、模型训练和评估。
    
    Attributes:
        graph: 输入的图结构。
        side_info: 补充信息字典。
        embedding_dim: 嵌入向量的维度。
        side_info_dim: 补充信息的维度。
        walk_length: 每次随机游走的长度。
        num_walks: 每个节点的随机游走次数。
        window_size: Skip-gram模型的窗口大小。
        learning_rate: 学习率。
        epochs: 训练的轮数。
    """
    def __init__(self, graph: nx.Graph, side_info: Dict[int, np.ndarray], embedding_dim: int, side_info_dim: int, walk_length: int, num_walks: int, window_size: int, learning_rate: float, epochs: int):
        self.graph = graph
        self.side_info = side_info
        self.embedding_dim = embedding_dim
        self.side_info_dim = side_info_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.vocab_size = len(graph.nodes)
        self.model = EGESModel(self.vocab_size, embedding_dim, side_info_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.walks = self._generate_walks()
        self.dataset = EGESDataset(self.walks, side_info, window_size)

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
        训练EGES模型。
        """
        data_loader = DataLoader(self.dataset, batch_size=128, shuffle=True)
        
        for epoch in range(self.epochs):
            total_loss = 0
            for target, context, side_info_vector in data_loader:
                self.optimizer.zero_grad()
                target = target.to(torch.int64)
                context = context.to(torch.int64)
                side_info_vector = side_info_vector.float()
                output = self.model(target, side_info_vector)
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
side_info = {i: np.random.randn(10) for i in graph.nodes}  # 示例补充信息

# 训练EGES模型
trainer = EGESTrainer(graph=graph, side_info=side_info, embedding_dim=128, side_info_dim=10, walk_length=10, num_walks=80, window_size=5, learning_rate=0.01, epochs=10)
trainer.train()

# 获取节点的嵌入向量
node_id = 0
embedding_vector = trainer.get_embedding(node_id)
print(f"Node {node_id} embedding vector: {embedding_vector}")

# 01_4.4.2 Node2vec——同质性和结构性的权衡

"""
Lecture: 第4章 Embedding技术在推荐系统中的应用/4.4 Graph Embedding——引入更多结构信息的图嵌入技术
Content: 01_4.4.2 Node2vec——同质性和结构性的权衡
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import networkx as nx
import random

class Node2VecDataset(Dataset):
    """
    Node2Vec数据集类，用于存储和提供训练数据。

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

class Node2VecModel(nn.Module):
    """
    Node2Vec模型类，通过Skip-gram模型实现节点嵌入。
    
    Attributes:
        embedding_dim: 嵌入向量的维度。
        vocab_size: 词汇表大小（节点数）。
        embeddings: 嵌入层，用于存储节点的嵌入向量。
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(Node2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_nodes: torch.Tensor) -> torch.Tensor:
        return self.embeddings(input_nodes)

class Node2VecTrainer:
    """
    Node2Vec训练器类，负责数据预处理、模型训练和评估。
    
    Attributes:
        graph: 输入的图结构。
        embedding_dim: 嵌入向量的维度。
        walk_length: 每次随机游走的长度。
        num_walks: 每个节点的随机游走次数。
        window_size: Skip-gram模型的窗口大小。
        p: 返回参数。
        q: 进出参数。
        learning_rate: 学习率。
        epochs: 训练的轮数。
    """
    def __init__(self, graph: nx.Graph, embedding_dim: int, walk_length: int, num_walks: int, window_size: int, p: float, q: float, learning_rate: float, epochs: int):
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.p = p
        self.q = q
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.vocab_size = len(graph.nodes)
        self.model = Node2VecModel(self.vocab_size, embedding_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.alias_nodes, self.alias_edges = self._preprocess_transition_probs()
        self.walks = self._generate_walks()
        self.dataset = Node2VecDataset(self.walks, window_size)

    def _preprocess_transition_probs(self) -> Tuple[Dict[int, np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
        """
        预处理转移概率。
        
        Returns:
            alias_nodes: 节点别名采样字典。
            alias_edges: 边别名采样字典。
        """
        alias_nodes = {}
        alias_edges = {}
        for node in self.graph.nodes:
            unnormalized_probs = [self.graph[node][nbr]['weight'] for nbr in self.graph.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = self._alias_setup(normalized_probs)

        for edge in self.graph.edges:
            alias_edges[edge] = self._get_alias_edge(edge[0], edge[1])
            if not self.graph.is_directed():
                alias_edges[(edge[1], edge[0])] = self._get_alias_edge(edge[1], edge[0])

        return alias_nodes, alias_edges

    def _alias_setup(self, probs: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        别名采样预处理。
        
        Args:
            probs: 概率列表。
        
        Returns:
            alias_table: 别名表。
            prob_table: 概率表。
        """
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)
        smaller = []
        larger = []

        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while smaller and larger:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q

    def _get_alias_edge(self, src: int, dst: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取边的别名采样。
        
        Args:
            src: 边的起始节点。
            dst: 边的终止节点。
        
        Returns:
            alias_edge: 边的别名表和概率表。
        """
        unnormalized_probs = []
        for dst_nbr in self.graph.neighbors(dst):
            if dst_nbr == src:
                unnormalized_probs.append(self.graph[dst][dst_nbr]['weight'] / self.p)
            elif self.graph.has_edge(dst_nbr, src):
                unnormalized_probs.append(self.graph[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(self.graph[dst][dst_nbr]['weight'] / self.q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        return self._alias_setup(normalized_probs)

    def _alias_draw(self, J: np.ndarray, q: np.ndarray) -> int:
        """
        别名采样。
        
        Args:
            J: 别名表。
            q: 概率表。
        
        Returns:
            idx: 采样结果索引。
        """
        K = len(J)
        kk = int(np.floor(np.random.rand() * K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]

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
                walk = self._node2vec_walk(node)
                walks.append(walk)
        return walks

    def _node2vec_walk(self, start_node: int) -> List[int]:
        """
        从指定节点开始进行Node2Vec随机游走。
        
        Args:
            start_node: 起始节点。
        
        Returns:
            walk: 随机游走生成的节点序列。
        """
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.graph.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[self._alias_draw(self.alias_nodes[cur][0], self.alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next_node = cur_nbrs[self._alias_draw(self.alias_edges[(prev, cur)][0], self.alias_edges[(prev, cur)][1])]
                    walk.append(next_node)
            else:
                break
        return walk

    def train(self):
        """
        训练Node2Vec模型。
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

# 训练Node2Vec模型
trainer = Node2VecTrainer(graph=graph, embedding_dim=128, walk_length=10, num_walks=80, window_size=5, p=1, q=1, learning_rate=0.01, epochs=10)
trainer.train()

# 获取节点的嵌入向量
node_id = 0
embedding_vector = trainer.get_embedding(node_id)
print(f"Node {node_id} embedding vector: {embedding_vector}")

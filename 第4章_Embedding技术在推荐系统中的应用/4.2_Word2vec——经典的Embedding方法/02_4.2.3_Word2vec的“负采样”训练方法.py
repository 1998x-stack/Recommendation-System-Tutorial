# 02_4.2.3 Word2vec的“负采样”训练方法

"""
Lecture: 第4章 Embedding技术在推荐系统中的应用/4.2 Word2vec——经典的Embedding方法
Content: 02_4.2.3 Word2vec的“负采样”训练方法
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from typing import List, Tuple

class Word2VecDataset(torch.utils.data.Dataset):
    """
    Word2Vec 数据集类，用于生成 Skip-Gram 训练样本。
    
    Attributes:
        data: 语料库列表
        word2idx: 词到索引的映射
        idx2word: 索引到词的映射
        pairs: Skip-Gram 样本对
        vocab_size: 词汇表大小
    """
    def __init__(self, corpus: List[str], window_size: int = 2) -> None:
        self.data = corpus
        self.window_size = window_size
        self.word2idx, self.idx2word = self._build_vocab(self.data)
        self.pairs = self._generate_pairs(self.data, self.window_size)
        self.vocab_size = len(self.word2idx)
        
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[int, int]:
        return self.pairs[idx]
    
    def _build_vocab(self, corpus: List[str]) -> Tuple[dict, dict]:
        """
        构建词汇表和索引映射。
        
        Args:
            corpus: 语料库列表
        
        Returns:
            word2idx: 词到索引的映射
            idx2word: 索引到词的映射
        """
        word_counts = Counter(corpus)
        idx2word = [word for word, _ in word_counts.items()]
        word2idx = {word: idx for idx, word in enumerate(idx2word)}
        return word2idx, idx2word
    
    def _generate_pairs(self, corpus: List[str], window_size: int) -> List[Tuple[int, int]]:
        """
        生成 Skip-Gram 样本对。
        
        Args:
            corpus: 语料库列表
            window_size: 窗口大小
        
        Returns:
            pairs: Skip-Gram 样本对列表
        """
        pairs = []
        for i, word in enumerate(corpus):
            for j in range(max(0, i - window_size), min(len(corpus), i + window_size + 1)):
                if i != j:
                    pairs.append((self.word2idx[word], self.word2idx[corpus[j]]))
        return pairs


class Word2VecModel(nn.Module):
    """
    Word2Vec 模型类，使用 Skip-Gram 结构和负采样训练方法。
    
    Attributes:
        embedding_dim: 嵌入向量维度
        vocab_size: 词汇表大小
        embeddings: 嵌入层
    """
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super(Word2VecModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, pos_words: torch.Tensor, neg_words: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算。
        
        Args:
            pos_words: 正样本词索引张量
            neg_words: 负样本词索引张量
        
        Returns:
            loss: 训练损失
        """
        pos_embedding = self.embeddings(pos_words)
        neg_embedding = self.embeddings(neg_words)
        
        pos_loss = -torch.log(torch.sigmoid(torch.sum(pos_embedding, dim=1)))
        neg_loss = -torch.log(torch.sigmoid(-torch.sum(neg_embedding, dim=1)))
        
        loss = torch.mean(pos_loss + neg_loss)
        return loss


def train_word2vec(corpus: List[str], embedding_dim: int = 100, window_size: int = 2, num_epochs: int = 5, batch_size: int = 64, neg_samples: int = 10) -> Word2VecModel:
    """
    训练 Word2Vec 模型。
    
    Args:
        corpus: 语料库列表
        embedding_dim: 嵌入向量维度
        window_size: 窗口大小
        num_epochs: 训练轮数
        batch_size: 批量大小
        neg_samples: 负样本数量
    
    Returns:
        model: 训练好的 Word2Vec 模型
    """
    dataset = Word2VecDataset(corpus, window_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = Word2VecModel(dataset.vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for pos_words, context_words in dataloader:
            neg_words = torch.randint(0, dataset.vocab_size, (batch_size, neg_samples))
            
            optimizer.zero_grad()
            loss = model(pos_words, neg_words)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch: {epoch+1}, Loss: {total_loss / len(dataloader)}')
    
    return model


if __name__ == "__main__":
    corpus = ["我", "爱", "自然", "语言", "处理", "和", "深度", "学习", "自然", "语言", "处理", "非常", "有趣"]
    model = train_word2vec(corpus, embedding_dim=50, window_size=2, num_epochs=10, batch_size=4, neg_samples=5)

    # 保存模型
    torch.save(model.state_dict(), "word2vec_model.pth")
    
    # 打印一些嵌入向量示例
    word_indices = [0, 1, 2, 3, 4]  # 示例单词索引
    embeddings = model.embeddings(torch.tensor(word_indices))
    for idx, embedding in zip(word_indices, embeddings):
        print(f"Word: {corpus[idx]}, Embedding: {embedding.detach().numpy()}")
# 01_2.2.2 用户相似度计算

"""
Lecture: 第2章 前深度学习时代——推荐系统的进化之路/2.2 协同过滤——经典的推荐算法
Content: 01_2.2.2 用户相似度计算
"""

import numpy as np
from typing import Dict, List, Tuple

class CollaborativeFiltering:
    def __init__(self, user_item_matrix: np.ndarray):
        """
        初始化协同过滤类

        Args:
            user_item_matrix (np.ndarray): 用户-物品评分矩阵
        """
        self.user_item_matrix = user_item_matrix

    def cosine_similarity(self, user1: int, user2: int) -> float:
        """
        计算两个用户之间的余弦相似度

        Args:
            user1 (int): 用户1的索引
            user2 (int): 用户2的索引

        Returns:
            float: 余弦相似度
        """
        vec1 = self.user_item_matrix[user1]
        vec2 = self.user_item_matrix[user2]
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def pearson_correlation(self, user1: int, user2: int) -> float:
        """
        计算两个用户之间的皮尔逊相关系数

        Args:
            user1 (int): 用户1的索引
            user2 (int): 用户2的索引

        Returns:
            float: 皮尔逊相关系数
        """
        vec1 = self.user_item_matrix[user1]
        vec2 = self.user_item_matrix[user2]
        mean1 = np.mean(vec1[vec1 > 0])
        mean2 = np.mean(vec2[vec2 > 0])
        centered_vec1 = vec1 - mean1
        centered_vec2 = vec2 - mean2
        mask = (vec1 > 0) & (vec2 > 0)
        if not np.any(mask):
            return 0.0
        centered_vec1 = centered_vec1[mask]
        centered_vec2 = centered_vec2[mask]
        dot_product = np.dot(centered_vec1, centered_vec2)
        norm1 = np.linalg.norm(centered_vec1)
        norm2 = np.linalg.norm(centered_vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def adjusted_cosine_similarity(self, user1: int, user2: int) -> float:
        """
        计算两个用户之间的修正余弦相似度

        Args:
            user1 (int): 用户1的索引
            user2 (int): 用户2的索引

        Returns:
            float: 修正余弦相似度
        """
        vec1 = self.user_item_matrix[user1]
        vec2 = self.user_item_matrix[user2]
        item_means = np.mean(self.user_item_matrix, axis=0)
        adjusted_vec1 = vec1 - item_means
        adjusted_vec2 = vec2 - item_means
        mask = (vec1 > 0) & (vec2 > 0)
        if not np.any(mask):
            return 0.0
        adjusted_vec1 = adjusted_vec1[mask]
        adjusted_vec2 = adjusted_vec2[mask]
        dot_product = np.dot(adjusted_vec1, adjusted_vec2)
        norm1 = np.linalg.norm(adjusted_vec1)
        norm2 = np.linalg.norm(adjusted_vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def compute_similarities(self, method: str = 'cosine') -> Dict[Tuple[int, int], float]:
        """
        计算所有用户之间的相似度

        Args:
            method (str): 相似度计算方法（'cosine', 'pearson', 'adjusted_cosine'）

        Returns:
            Dict[Tuple[int, int], float]: 用户对之间的相似度字典
        """
        num_users = self.user_item_matrix.shape[0]
        similarities = {}
        for user1 in range(num_users):
            for user2 in range(user1 + 1, num_users):
                if method == 'cosine':
                    similarity = self.cosine_similarity(user1, user2)
                elif method == 'pearson':
                    similarity = self.pearson_correlation(user1, user2)
                elif method == 'adjusted_cosine':
                    similarity = self.adjusted_cosine_similarity(user1, user2)
                else:
                    raise ValueError("Invalid method: choose from 'cosine', 'pearson', 'adjusted_cosine'")
                similarities[(user1, user2)] = similarity
                similarities[(user2, user1)] = similarity
        return similarities

    def predict_rating(self, user: int, item: int, similarities: Dict[Tuple[int, int], float], top_n: int = 10) -> float:
        """
        预测用户对物品的评分

        Args:
            user (int): 用户索引
            item (int): 物品索引
            similarities (Dict[Tuple[int, int], float]): 用户对之间的相似度字典
            top_n (int): 使用相似用户的数量

        Returns:
            float: 预测评分
        """
        similar_users = sorted(similarities.keys(), key=lambda x: similarities[x], reverse=True)
        top_similar_users = [u for u in similar_users if u[0] == user or u[1] == user][:top_n]
        numerator = sum(similarities[(u, user)] * self.user_item_matrix[u, item] for u in top_similar_users)
        denominator = sum(abs(similarities[(u, user)]) for u in top_similar_users)
        if denominator == 0:
            return 0.0
        return numerator / denominator

def main():
    # 示例用户-物品评分矩阵
    user_item_matrix = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])

    # 初始化协同过滤模型
    cf = CollaborativeFiltering(user_item_matrix)

    # 计算所有用户之间的余弦相似度
    cosine_similarities = cf.compute_similarities(method='cosine')
    print("Cosine Similarities:")
    for pair, similarity in cosine_similarities.items():
        print(f"User {pair[0]} - User {pair[1]}: {similarity}")

    # 计算所有用户之间的皮尔逊相关系数
    pearson_similarities = cf.compute_similarities(method='pearson')
    print("\nPearson Correlation Coefficients:")
    for pair, similarity in pearson_similarities.items():
        print(f"User {pair[0]} - User {pair[1]}: {similarity}")

    # 计算所有用户之间的修正余弦相似度
    adjusted_cosine_similarities = cf.compute_similarities(method='adjusted_cosine')
    print("\nAdjusted Cosine Similarities:")
    for pair, similarity in adjusted_cosine_similarities.items():
        print(f"User {pair[0]} - User {pair[1]}: {similarity}")

    # 预测用户0对物品2的评分
    predicted_rating = cf.predict_rating(user=0, item=2, similarities=cosine_similarities, top_n=2)
    print(f"\nPredicted Rating for User 0 on Item 2: {predicted_rating}")

if __name__ == "__main__":
    main()

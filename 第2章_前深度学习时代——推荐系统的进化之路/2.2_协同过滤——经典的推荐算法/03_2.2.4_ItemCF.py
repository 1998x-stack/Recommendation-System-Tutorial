# 03_2.2.4 ItemCF

"""
Lecture: 第2章 前深度学习时代——推荐系统的进化之路/2.2 协同过滤——经典的推荐算法
Content: 03_2.2.4 ItemCF
"""

import numpy as np
from typing import Dict, List, Tuple

class ItemCollaborativeFiltering:
    def __init__(self, user_item_matrix: np.ndarray):
        """
        初始化基于物品的协同过滤类

        Args:
            user_item_matrix (np.ndarray): 用户-物品评分矩阵
        """
        self.user_item_matrix = user_item_matrix
        self.item_similarity_matrix = self.compute_item_similarities()

    def compute_item_similarities(self) -> np.ndarray:
        """
        计算所有物品之间的相似度矩阵

        Returns:
            np.ndarray: 物品相似度矩阵
        """
        num_items = self.user_item_matrix.shape[1]
        item_similarity_matrix = np.zeros((num_items, num_items))

        for item1 in range(num_items):
            for item2 in range(item1 + 1, num_items):
                similarity = self.cosine_similarity(item1, item2)
                item_similarity_matrix[item1, item2] = similarity
                item_similarity_matrix[item2, item1] = similarity

        return item_similarity_matrix

    def cosine_similarity(self, item1: int, item2: int) -> float:
        """
        计算两个物品之间的余弦相似度

        Args:
            item1 (int): 物品1的索引
            item2 (int): 物品2的索引

        Returns:
            float: 余弦相似度
        """
        vec1 = self.user_item_matrix[:, item1]
        vec2 = self.user_item_matrix[:, item2]
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def predict_rating(self, user: int, item: int, top_n: int = 10) -> float:
        """
        预测用户对物品的评分

        Args:
            user (int): 用户索引
            item (int): 物品索引
            top_n (int): 使用相似物品的数量

        Returns:
            float: 预测评分
        """
        similar_items = sorted([(other, self.item_similarity_matrix[item, other])
                                for other in range(self.user_item_matrix.shape[1]) if other != item],
                               key=lambda x: x[1], reverse=True)[:top_n]
        numerator = sum(sim * self.user_item_matrix[user, other] for other, sim in similar_items if self.user_item_matrix[user, other] > 0)
        denominator = sum(abs(sim) for other, sim in similar_items if self.user_item_matrix[user, other] > 0)
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def generate_recommendations(self, user: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        为用户生成推荐列表

        Args:
            user (int): 用户索引
            top_k (int): 推荐的物品数量

        Returns:
            List[Tuple[int, float]]: 推荐的物品列表和预测评分
        """
        user_ratings = self.user_item_matrix[user]
        predicted_ratings = []
        for item in range(user_ratings.shape[0]):
            if user_ratings[item] == 0:  # 仅对未评分的物品进行预测
                predicted_rating = self.predict_rating(user, item)
                predicted_ratings.append((item, predicted_rating))
        # 对预测评分进行排序并返回Top K的物品
        recommended_items = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:top_k]
        return recommended_items

def main():
    # 示例用户-物品评分矩阵
    user_item_matrix = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])

    # 初始化基于物品的协同过滤模型
    item_cf = ItemCollaborativeFiltering(user_item_matrix)

    # 为用户0生成推荐列表
    recommendations = item_cf.generate_recommendations(user=0, top_k=3)
    print(f"Recommendations for User 0: {recommendations}")

if __name__ == "__main__":
    main()

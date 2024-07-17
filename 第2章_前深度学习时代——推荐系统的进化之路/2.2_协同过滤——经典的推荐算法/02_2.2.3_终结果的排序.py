# 02_2.2.3 终结果的排序

"""
Lecture: 第2章 前深度学习时代——推荐系统的进化之路/2.2 协同过滤——经典的推荐算法
Content: 02_2.2.3 终结果的排序
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

    def compute_similarities(self, method: str = 'cosine') -> Dict[Tuple[int, int], float]:
        """
        计算所有用户之间的相似度

        Args:
            method (str): 相似度计算方法（'cosine'）

        Returns:
            Dict[Tuple[int, int], float]: 用户对之间的相似度字典
        """
        num_users = self.user_item_matrix.shape[0]
        similarities = {}
        for user1 in range(num_users):
            for user2 in range(user1 + 1, num_users):
                if method == 'cosine':
                    similarity = self.cosine_similarity(user1, user2)
                else:
                    raise ValueError("Invalid method: choose 'cosine'")
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
        similar_users = sorted([(other, sim) for (u1, other), sim in similarities.items() if u1 == user], key=lambda x: x[1], reverse=True)[:top_n]
        numerator = sum(sim * self.user_item_matrix[other, item] for other, sim in similar_users if self.user_item_matrix[other, item] > 0)
        denominator = sum(abs(sim) for other, sim in similar_users if self.user_item_matrix[other, item] > 0)
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
        similarities = self.compute_similarities(method='cosine')
        user_ratings = self.user_item_matrix[user]
        predicted_ratings = []
        for item in range(user_ratings.shape[0]):
            if user_ratings[item] == 0:  # 仅对未评分的物品进行预测
                predicted_rating = self.predict_rating(user, item, similarities)
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

    # 初始化协同过滤模型
    cf = CollaborativeFiltering(user_item_matrix)

    # 为用户0生成推荐列表
    recommendations = cf.generate_recommendations(user=0, top_k=3)
    print(f"Recommendations for User 0: {recommendations}")

if __name__ == "__main__":
    main()

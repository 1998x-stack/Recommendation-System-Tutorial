# 01_7.1.2 离线评估的指标

"""
Lecture: 第7章 推荐系统的评估/7.1 离线评估方法与基本评价指标
Content: 01_7.1.2 离线评估的指标
"""

from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np

class OfflineEvaluationMetrics:
    def __init__(self):
        pass

    def precision(self, true_positive: int, false_positive: int) -> float:
        """
        计算精确率

        Args:
            true_positive (int): 真正例数
            false_positive (int): 假正例数

        Returns:
            float: 精确率
        """
        if true_positive + false_positive == 0:
            return 0.0
        return true_positive / (true_positive + false_positive)

    def recall(self, true_positive: int, false_negative: int) -> float:
        """
        计算召回率

        Args:
            true_positive (int): 真正例数
            false_negative (int): 假负例数

        Returns:
            float: 召回率
        """
        if true_positive + false_negative == 0:
            return 0.0
        return true_positive / (true_positive + false_negative)

    def f1_score(self, precision: float, recall: float) -> float:
        """
        计算F1值

        Args:
            precision (float): 精确率
            recall (float): 召回率

        Returns:
            float: F1值
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def mean_rank(self, ranks: List[int]) -> float:
        """
        计算平均排名

        Args:
            ranks (List[int]): 推荐结果中相关项目的排名列表

        Returns:
            float: 平均排名
        """
        if not ranks:
            return 0.0
        return np.mean(ranks)

    def mean_reciprocal_rank(self, ranks: List[int]) -> float:
        """
        计算平均倒数排名

        Args:
            ranks (List[int]): 推荐结果中相关项目的排名列表

        Returns:
            float: 平均倒数排名
        """
        if not ranks:
            return 0.0
        return np.mean([1.0 / rank for rank in ranks])

    def coverage(self, recommended_items: List[int], all_items: List[int]) -> float:
        """
        计算覆盖率

        Args:
            recommended_items (List[int]): 推荐的项目列表
            all_items (List[int]): 所有项目列表

        Returns:
            float: 覆盖率
        """
        if not all_items:
            return 0.0
        return len(set(recommended_items)) / len(all_items)

    def diversity(self, recommended_items: List[int], similarity_matrix: Dict[Tuple[int, int], float]) -> float:
        """
        计算多样性

        Args:
            recommended_items (List[int]): 推荐的项目列表
            similarity_matrix (Dict[Tuple[int, int], float]): 项目相似度矩阵

        Returns:
            float: 多样性
        """
        if len(recommended_items) < 2:
            return 0.0
        pairwise_similarities = [
            similarity_matrix[(min(i, j), max(i, j))]
            for idx, i in enumerate(recommended_items)
            for j in recommended_items[idx + 1:]
        ]
        if not pairwise_similarities:
            return 1.0
        return 1.0 - np.mean(pairwise_similarities)

    def novelty(self, recommended_items: List[int], popularity_dict: Dict[int, float]) -> float:
        """
        计算新颖性

        Args:
            recommended_items (List[int]): 推荐的项目列表
            popularity_dict (Dict[int, float]): 项目流行度字典

        Returns:
            float: 新颖性
        """
        if not recommended_items:
            return 0.0
        return 1.0 - np.mean([popularity_dict[item] for item in recommended_items])

# 示例使用
evaluation_metrics = OfflineEvaluationMetrics()

# 精确率和召回率
precision_value = evaluation_metrics.precision(true_positive=50, false_positive=10)
recall_value = evaluation_metrics.recall(true_positive=50, false_negative=20)

# F1值
f1_value = evaluation_metrics.f1_score(precision=precision_value, recall=recall_value)

# 平均排名和平均倒数排名
mean_rank_value = evaluation_metrics.mean_rank(ranks=[1, 3, 5, 7])
mean_rr_value = evaluation_metrics.mean_reciprocal_rank(ranks=[1, 3, 5, 7])

# 覆盖率
coverage_value = evaluation_metrics.coverage(recommended_items=[1, 2, 3, 4], all_items=list(range(1, 101)))

# 多样性
similarity_matrix = {
    (1, 2): 0.9, (1, 3): 0.2, (1, 4): 0.4,
    (2, 3): 0.5, (2, 4): 0.3,
    (3, 4): 0.7
}
diversity_value = evaluation_metrics.diversity(recommended_items=[1, 2, 3, 4], similarity_matrix=similarity_matrix)

# 新颖性
popularity_dict = {
    1: 0.9, 2: 0.8, 3: 0.1, 4: 0.4
}
novelty_value = evaluation_metrics.novelty(recommended_items=[1, 2, 3, 4], popularity_dict=popularity_dict)

print(f"Precision: {precision_value}")
print(f"Recall: {recall_value}")
print(f"F1 Score: {f1_value}")
print(f"Mean Rank: {mean_rank_value}")
print(f"Mean Reciprocal Rank: {mean_rr_value}")
print(f"Coverage: {coverage_value}")
print(f"Diversity: {diversity_value}")
print(f"Novelty: {novelty_value}")

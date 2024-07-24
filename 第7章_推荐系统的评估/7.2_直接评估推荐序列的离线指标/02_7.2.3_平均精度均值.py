# 02_7.2.3 平均精度均值

"""
Lecture: 第7章 推荐系统的评估/7.2 直接评估推荐序列的离线指标
Content: 02_7.2.3 平均精度均值
"""

import numpy as np
from typing import List, Tuple


class MeanAveragePrecision:
    """
    平均精度均值类
    
    该类用于计算信息检索和推荐系统的平均精度均值（MAP）。
    """

    def __init__(self, y_trues: List[List[int]], y_scores: List[List[float]]) -> None:
        """
        初始化MeanAveragePrecision类
        
        Args:
            y_trues (List[List[int]]): 多个查询或推荐结果的实际标签列表
            y_scores (List[List[float]]): 多个查询或推荐结果的预测得分列表
        """
        self.y_trues = [np.array(y_true) for y_true in y_trues]
        self.y_scores = [np.array(y_score) for y_score in y_scores]

    def _calculate_average_precision(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        计算单个查询或推荐结果的平均精度（AP）
        
        Args:
            y_true (np.ndarray): 实际标签
            y_score (np.ndarray): 预测得分
        
        Returns:
            float: 平均精度
        """
        # 按得分排序
        sorted_indices = np.argsort(-y_score)
        y_true = y_true[sorted_indices]

        # 计算精度和召回率
        precisions = []
        relevant_count = 0
        for i, label in enumerate(y_true):
            if label == 1:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))
        
        if relevant_count == 0:
            return 0.0
        
        return np.mean(precisions)

    def calculate_map(self) -> float:
        """
        计算多个查询或推荐结果的平均精度均值（MAP）
        
        Returns:
            float: 平均精度均值
        """
        average_precisions = []
        for y_true, y_score in zip(self.y_trues, self.y_scores):
            ap = self._calculate_average_precision(y_true, y_score)
            average_precisions.append(ap)
        
        return np.mean(average_precisions)

    def print_map(self) -> None:
        """
        打印MAP值
        """
        map_value = self.calculate_map()
        print(f"平均精度均值 (MAP): {map_value:.4f}")


# 示例使用
if __name__ == "__main__":
    # 实际标签
    y_trues = [
        [1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 0, 0, 1, 1]
    ]

    # 预测得分
    y_scores = [
        [0.9, 0.2, 0.8, 0.6, 0.3],
        [0.1, 0.7, 0.5, 0.4, 0.6],
        [0.6, 0.1, 0.4, 0.8, 0.7]
    ]

    # 创建MeanAveragePrecision实例
    map_calculator = MeanAveragePrecision(y_trues, y_scores)

    # 计算并打印MAP值
    map_calculator.print_map()
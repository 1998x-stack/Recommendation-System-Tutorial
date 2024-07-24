# 00_7.2.1 P-R曲线

"""
Lecture: 第7章 推荐系统的评估/7.2 直接评估推荐序列的离线指标
Content: 00_7.2.1 P-R曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class PrecisionRecallCurve:
    """
    Precision-Recall Curve Class
    
    该类用于计算和绘制精确率-召回率曲线（P-R曲线）。
    """

    def __init__(self, y_true: List[int], y_scores: List[float]) -> None:
        """
        初始化PrecisionRecallCurve类
        
        Args:
            y_true (List[int]): 实际标签列表
            y_scores (List[float]): 预测得分列表
        """
        self.y_true = np.array(y_true)
        self.y_scores = np.array(y_scores)
        self.precision_recall_points = self._calculate_precision_recall()

    def _calculate_precision_recall(self) -> List[Tuple[float, float]]:
        """
        计算不同阈值下的精确率和召回率
        
        Returns:
            List[Tuple[float, float]]: 精确率和召回率的列表
        """
        thresholds = np.sort(self.y_scores)[::-1]
        precision_recall_points = []

        for threshold in thresholds:
            y_pred = self.y_scores >= threshold
            tp = np.sum((self.y_true == 1) & (y_pred == 1))
            fp = np.sum((self.y_true == 0) & (y_pred == 1))
            fn = np.sum((self.y_true == 1) & (y_pred == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (fn + tp) if (fn + tp) > 0 else 0.0

            precision_recall_points.append((precision, recall))

        return precision_recall_points

    def plot_precision_recall_curve(self) -> None:
        """
        绘制精确率-召回率曲线
        
        """
        precisions, recalls = zip(*self.precision_recall_points)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker='.')
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title('精确率-召回率曲线 (P-R Curve)')
        plt.grid(True)
        plt.show()

    def calculate_auc(self) -> float:
        """
        计算P-R曲线下的面积（AUC）
        
        Returns:
            float: AUC值
        """
        precisions, recalls = zip(*self.precision_recall_points)
        return np.trapz(precisions, recalls)


# 示例使用
if __name__ == "__main__":
    # 实际标签
    y_true = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]

    # 预测得分
    y_scores = [0.1, 0.4, 0.35, 0.8, 0.45, 0.2, 0.5, 0.9, 0.7, 0.3]

    # 创建PrecisionRecallCurve实例
    pr_curve = PrecisionRecallCurve(y_true, y_scores)

    # 打印计算的精确率和召回率
    print("精确率和召回率: ", pr_curve.precision_recall_points)

    # 绘制P-R曲线
    pr_curve.plot_precision_recall_curve()

    # 计算并打印AUC值
    auc_value = pr_curve.calculate_auc()
    print(f"P-R曲线下的面积 (AUC): {auc_value:.4f}")
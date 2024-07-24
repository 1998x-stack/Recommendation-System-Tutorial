# 01_7.2.2 ROC曲线

"""
Lecture: 第7章 推荐系统的评估/7.2 直接评估推荐序列的离线指标
Content: 01_7.2.2 ROC曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class ROCCurve:
    """
    ROC曲线类
    
    该类用于计算和绘制受试者工作特征曲线（ROC曲线）。
    """

    def __init__(self, y_true: List[int], y_scores: List[float]) -> None:
        """
        初始化ROCCurve类
        
        Args:
            y_true (List[int]): 实际标签列表
            y_scores (List[float]): 预测得分列表
        """
        self.y_true = np.array(y_true)
        self.y_scores = np.array(y_scores)
        self.roc_points = self._calculate_roc_points()

    def _calculate_roc_points(self) -> List[Tuple[float, float]]:
        """
        计算不同阈值下的真阳性率和假阳性率
        
        Returns:
            List[Tuple[float, float]]: 真阳性率和假阳性率的列表
        """
        thresholds = np.sort(self.y_scores)[::-1]
        roc_points = []

        for threshold in thresholds:
            y_pred = self.y_scores >= threshold
            tp = np.sum((self.y_true == 1) & (y_pred == 1))
            fp = np.sum((self.y_true == 0) & (y_pred == 1))
            fn = np.sum((self.y_true == 1) & (y_pred == 0))
            tn = np.sum((self.y_true == 0) & (y_pred == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            roc_points.append((fpr, tpr))

        return roc_points

    def plot_roc_curve(self) -> None:
        """
        绘制ROC曲线
        
        """
        fprs, tprs = zip(*self.roc_points)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fprs, tprs, marker='.')
        plt.xlabel('假阳性率 (False Positive Rate)', fontproperties=my_font)
        plt.ylabel('真阳性率 (True Positive Rate)', fontproperties=my_font)
        plt.title('ROC曲线', fontproperties=my_font)
        plt.plot([0, 1], [0, 1], 'k--')  # 添加对角线
        plt.grid(True)
        plt.show()

    def calculate_auc(self) -> float:
        """
        计算ROC曲线下的面积（AUC）
        
        Returns:
            float: AUC值
        """
        fprs, tprs = zip(*self.roc_points)
        return np.trapz(tprs, fprs)


# 示例使用
if __name__ == "__main__":
    # 实际标签
    y_true = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]

    # 预测得分
    y_scores = [0.1, 0.4, 0.35, 0.8, 0.45, 0.2, 0.5, 0.9, 0.7, 0.3]

    # 创建ROCCurve实例
    roc_curve = ROCCurve(y_true, y_scores)

    # 打印计算的真阳性率和假阳性率
    print("真阳性率和假阳性率: ", roc_curve.roc_points)

    # 绘制ROC曲线
    roc_curve.plot_roc_curve()

    # 计算并打印AUC值
    auc_value = roc_curve.calculate_auc()
    print(f"ROC曲线下的面积 (AUC): {auc_value:.4f}")
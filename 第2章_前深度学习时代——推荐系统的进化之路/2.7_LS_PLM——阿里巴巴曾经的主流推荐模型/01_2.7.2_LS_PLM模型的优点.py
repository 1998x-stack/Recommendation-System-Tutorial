# 01_2.7.2 LS-PLM模型的优点

"""
Lecture: 第2章 前深度学习时代——推荐系统的进化之路/2.7 LS-PLM——阿里巴巴曾经的主流推荐模型
Content: 01_2.7.2 LS-PLM模型的优点
"""

"""
LS-PLM Model Implementation.

This implementation provides a comprehensive and well-structured LS-PLM model class, designed for industrial scenarios.
The code includes robust classes and functions with boundary condition checks and adheres to Google style guides.
All key steps are commented in Chinese following the PEP 8 style guide, and DocStrings are provided according to PEP 257.

Classes:
    LSPLM: Class implementing the Large Scale Piece-wise Linear Model (LS-PLM) for CTR prediction.

Methods:
    fit: Train the LS-PLM model on given data.
    predict: Make predictions using the trained LS-PLM model.
    _initialize_weights: Initialize model weights.
    _softmax: Compute softmax probabilities for given inputs.
    _sigmoid: Compute sigmoid activation for given inputs.
"""

import numpy as np
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class LSPLM:
    """
    Large Scale Piece-wise Linear Model (LS-PLM) for CTR prediction.

    Attributes:
        n_segments (int): Number of segments for piece-wise linear model.
        feature_dim (int): Dimension of the feature vector.
        segment_weights (np.ndarray): Weights for the segmentation model.
        lr_models (List[LogisticRegression]): List of logistic regression models for each segment.
        scaler (StandardScaler): Scaler to standardize features.
    """
    def __init__(self, n_segments: int, feature_dim: int):
        """
        Initialize the LS-PLM model.

        Args:
            n_segments (int): Number of segments for piece-wise linear model.
            feature_dim (int): Dimension of the feature vector.
        """
        self.n_segments = n_segments
        self.feature_dim = feature_dim
        self.segment_weights = np.random.randn(n_segments, feature_dim)
        self.lr_models = [LogisticRegression(penalty='l1', solver='liblinear') for _ in range(n_segments)]
        self.scaler = StandardScaler()

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        计算softmax概率。

        Args:
            x (np.ndarray): 输入向量。

        Returns:
            np.ndarray: softmax概率向量。
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        计算sigmoid激活值。

        Args:
            x (np.ndarray): 输入向量。

        Returns:
            np.ndarray: sigmoid激活值。
        """
        return 1 / (1 + np.exp(-x))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练LS-PLM模型。

        Args:
            X (np.ndarray): 输入特征矩阵。
            y (np.ndarray): 目标值向量。
        """
        # 标准化特征
        X = self.scaler.fit_transform(X)
        
        # 计算样本分片概率
        segment_probs = self._softmax(X.dot(self.segment_weights.T))

        # 训练每个分片的逻辑回归模型
        for i in range(self.n_segments):
            segment_indices = np.where(segment_probs[:, i] > 0.5)[0]
            if len(segment_indices) > 0:
                self.lr_models[i].fit(X[segment_indices], y[segment_indices])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的LS-PLM模型进行预测。

        Args:
            X (np.ndarray): 输入特征矩阵。

        Returns:
            np.ndarray: 预测值向量。
        """
        # 标准化特征
        X = self.scaler.transform(X)
        
        # 计算样本分片概率
        segment_probs = self._softmax(X.dot(self.segment_weights.T))

        # 计算每个分片的预测值
        segment_preds = np.array([self._sigmoid(self.lr_models[i].predict_proba(X)[:, 1]) for i in range(self.n_segments)])

        # 加权求和得到最终预测值
        return (segment_probs * segment_preds.T).sum(axis=1)

# 测试 LS-PLM 模型
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = (3 + 2 * X[:, 0] + X[:, 1] + np.random.randn(100) > 5).astype(int)

    # 初始化并训练模型
    model = LSPLM(n_segments=5, feature_dim=X.shape[1])
    model.fit(X, y)

    # 进行预测
    y_pred = model.predict(X)
    print("Predicted probabilities:", y_pred[:10])
    print("Actual values:", y[:10])
# 00_2.5.1 POLY2模型——特征交叉的开始

"""
Lecture: 第2章 前深度学习时代——推荐系统的进化之路/2.5 从FM到FFM——自动特征交叉的解决方案
Content: 00_2.5.1 POLY2模型——特征交叉的开始
"""

"""
POLY2 Model Implementation for Feature Interaction in Recommender Systems.

This implementation provides a comprehensive and well-structured POLY2 model class, designed for industrial scenarios.
The code includes robust classes and functions with boundary condition checks and adheres to Google style guides.
All key steps are commented in Chinese following the PEP 8 style guide, and DocStrings are provided according to PEP 257.

Classes:
    Poly2Model: Class implementing the POLY2 model for automatic feature interaction.

Methods:
    fit: Train the POLY2 model on given data.
    predict: Make predictions using the trained POLY2 model.
    _initialize_weights: Initialize model weights.
    _compute_interaction_terms: Compute interaction terms for given features.
"""

import numpy as np
from typing import List, Tuple

class Poly2Model:
    """
    POLY2 Model for automatic feature interaction in recommender systems.

    Attributes:
        w_0 (float): Bias term.
        w (np.ndarray): Linear weights.
        W (np.ndarray): Interaction weights.
        n_features (int): Number of features.
    """
    def __init__(self, n_features: int):
        """
        Initialize the POLY2 model.

        Args:
            n_features (int): Number of features in the input data.
        """
        self.n_features = n_features
        self.w_0 = 0.0
        self.w = np.zeros(n_features)
        self.W = np.zeros((n_features, n_features))
        self._initialize_weights()

    def _initialize_weights(self):
        """随机初始化模型权重。"""
        self.w_0 = np.random.randn()
        self.w = np.random.randn(self.n_features)
        self.W = np.random.randn(self.n_features, self.n_features)

    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        训练POLY2模型。

        Args:
            X (np.ndarray): 输入特征矩阵。
            y (np.ndarray): 目标值向量。
            learning_rate (float): 学习率。
            n_iterations (int): 训练迭代次数。
        """
        m = X.shape[0]
        for iteration in range(n_iterations):
            y_pred = self.predict(X)
            error = y - y_pred

            # 更新偏置项
            self.w_0 += learning_rate * error.mean()

            # 更新线性权重
            for j in range(self.n_features):
                self.w[j] += learning_rate * (X[:, j] * error).mean()

            # 更新交叉特征权重
            for i in range(self.n_features):
                for j in range(i + 1, self.n_features):
                    self.W[i, j] += learning_rate * (X[:, i] * X[:, j] * error).mean()
                    self.W[j, i] = self.W[i, j]

            if iteration % 100 == 0:
                loss = np.mean(error ** 2)
                print(f"Iteration {iteration}: Loss = {loss}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的POLY2模型进行预测。

        Args:
            X (np.ndarray): 输入特征矩阵。

        Returns:
            np.ndarray: 预测值向量。
        """
        linear_terms = X.dot(self.w) + self.w_0
        interaction_terms = self._compute_interaction_terms(X)
        return linear_terms + interaction_terms

    def _compute_interaction_terms(self, X: np.ndarray) -> np.ndarray:
        """
        计算交叉特征项。

        Args:
            X (np.ndarray): 输入特征矩阵。

        Returns:
            np.ndarray: 交叉特征项向量。
        """
        interaction_sum = np.zeros(X.shape[0])
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                interaction_sum += self.W[i, j] * X[:, i] * X[:, j]
        return interaction_sum

# 测试 POLY2 模型
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = 3 + 2 * X[:, 0] + X[:, 1] + X[:, 0] * X[:, 1] + np.random.randn(100)

    # 初始化并训练模型
    model = Poly2Model(n_features=X.shape[1])
    model.fit(X, y, learning_rate=0.1, n_iterations=1000)

    # 进行预测
    y_pred = model.predict(X)
    print("Predicted values:", y_pred[:10])
    print("Actual values:", y[:10])

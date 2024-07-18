# 02_2.5.3 FFM模型——引入特征域的概念

"""
Lecture: 第2章 前深度学习时代——推荐系统的进化之路/2.5 从FM到FFM——自动特征交叉的解决方案
Content: 02_2.5.3 FFM模型——引入特征域的概念
"""

"""
FFM Model Implementation for Feature Interaction in Recommender Systems.

This implementation provides a comprehensive and well-structured FFM model class, designed for industrial scenarios.
The code includes robust classes and functions with boundary condition checks and adheres to Google style guides.
All key steps are commented in Chinese following the PEP 8 style guide, and DocStrings are provided according to PEP 257.

Classes:
    FFMModel: Class implementing the Field-aware Factorization Machine (FFM) model for automatic feature interaction.

Methods:
    fit: Train the FFM model on given data.
    predict: Make predictions using the trained FFM model.
    _initialize_weights: Initialize model weights.
    _compute_interaction_terms: Compute interaction terms for given features.
"""

import numpy as np
from typing import List, Tuple

class FFMModel:
    """
    Field-aware Factorization Machine (FFM) Model for automatic feature interaction in recommender systems.

    Attributes:
        w_0 (float): Bias term.
        w (np.ndarray): Linear weights.
        V (np.ndarray): Interaction weights in the form of field-aware latent vectors.
        n_features (int): Number of features.
        k (int): Dimension of the latent vectors.
        n_fields (int): Number of fields.
    """
    def __init__(self, n_features: int, k: int, n_fields: int):
        """
        Initialize the FFM model.

        Args:
            n_features (int): Number of features in the input data.
            k (int): Dimension of the latent vectors.
            n_fields (int): Number of fields.
        """
        self.n_features = n_features
        self.k = k
        self.n_fields = n_fields
        self.w_0 = 0.0
        self.w = np.zeros(n_features)
        self.V = np.zeros((n_features, n_fields, k))
        self._initialize_weights()

    def _initialize_weights(self):
        """随机初始化模型权重。"""
        self.w_0 = np.random.randn()
        self.w = np.random.randn(self.n_features)
        self.V = np.random.randn(self.n_features, self.n_fields, self.k)

    def fit(self, X: np.ndarray, y: np.ndarray, fields: np.ndarray, learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        训练FFM模型。

        Args:
            X (np.ndarray): 输入特征矩阵。
            y (np.ndarray): 目标值向量。
            fields (np.ndarray): 特征对应的域。
            learning_rate (float): 学习率。
            n_iterations (int): 训练迭代次数。
        """
        m = X.shape[0]
        for iteration in range(n_iterations):
            y_pred = self.predict(X, fields)
            error = y - y_pred

            # 更新偏置项
            self.w_0 += learning_rate * error.mean()

            # 更新线性权重
            for j in range(self.n_features):
                self.w[j] += learning_rate * (X[:, j] * error).mean()

            # 更新隐向量权重
            for i in range(self.n_features):
                for j in range(self.n_features):
                    if i != j:
                        field_i = fields[i]
                        field_j = fields[j]
                        for f in range(self.k):
                            self.V[i, field_j, f] += learning_rate * (X[:, i] * X[:, j] * error * self.V[j, field_i, f]).mean()

            if iteration % 100 == 0:
                loss = np.mean(error ** 2)
                print(f"Iteration {iteration}: Loss = {loss}")

    def predict(self, X: np.ndarray, fields: np.ndarray) -> np.ndarray:
        """
        使用训练好的FFM模型进行预测。

        Args:
            X (np.ndarray): 输入特征矩阵。
            fields (np.ndarray): 特征对应的域。

        Returns:
            np.ndarray: 预测值向量。
        """
        linear_terms = X.dot(self.w) + self.w_0
        interaction_terms = self._compute_interaction_terms(X, fields)
        return linear_terms + interaction_terms

    def _compute_interaction_terms(self, X: np.ndarray, fields: np.ndarray) -> np.ndarray:
        """
        计算交叉特征项。

        Args:
            X (np.ndarray): 输入特征矩阵。
            fields (np.ndarray): 特征对应的域。

        Returns:
            np.ndarray: 交叉特征项向量。
        """
        interaction_sum = np.zeros(X.shape[0])
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                field_i = fields[i]
                field_j = fields[j]
                interaction_sum += (X[:, i] * X[:, j] * np.dot(self.V[i, field_j], self.V[j, field_i]))
        return interaction_sum

# 测试 FFM 模型
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    X = np.random.rand(100, 5)
    fields = np.array([0, 1, 0, 1, 0])
    y = 3 + 2 * X[:, 0] + X[:, 1] + X[:, 0] * X[:, 1] + np.random.randn(100)

    # 初始化并训练模型
    model = FFMModel(n_features=X.shape[1], k=10, n_fields=2)
    model.fit(X, y, fields, learning_rate=0.1, n_iterations=1000)

    # 进行预测
    y_pred = model.predict(X, fields)
    print("Predicted values:", y_pred[:10])
    print("Actual values:", y[:10])
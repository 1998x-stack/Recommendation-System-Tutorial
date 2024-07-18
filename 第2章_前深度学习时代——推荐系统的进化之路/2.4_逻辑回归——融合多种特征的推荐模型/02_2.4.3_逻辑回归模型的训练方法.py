# 02_2.4.3 逻辑回归模型的训练方法

"""
Lecture: 第2章 前深度学习时代——推荐系统的进化之路/2.4 逻辑回归——融合多种特征的推荐模型
Content: 02_2.4.3 逻辑回归模型的训练方法
"""

import numpy as np
from typing import Tuple, List

class LogisticRegressionModel:
    def __init__(self, learning_rate: float, iterations: int):
        """
        初始化逻辑回归模型

        Args:
            learning_rate (float): 学习率
            iterations (int): 迭代次数
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid激活函数

        Args:
            z (np.ndarray): 输入值

        Returns:
            np.ndarray: Sigmoid函数的输出值
        """
        return 1 / (1 + np.exp(-z))

    def loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        计算逻辑回归的损失函数（对数似然损失）

        Args:
            y (np.ndarray): 真实标签
            y_hat (np.ndarray): 预测标签

        Returns:
            float: 损失值
        """
        m = y.shape[0]
        return -1 / m * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def gradient_descent(self, X: np.ndarray, y: np.ndarray):
        """
        使用梯度下降法优化模型参数

        Args:
            X (np.ndarray): 特征矩阵
            y (np.ndarray): 标签向量
        """
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for i in range(self.iterations):
            model = np.dot(X, self.weights) + self.bias
            y_hat = self.sigmoid(model)

            dw = 1 / m * np.dot(X.T, (y_hat - y))
            db = 1 / m * np.sum(y_hat - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if (i + 1) % 100 == 0:
                loss = self.loss(y, y_hat)
                print(f"Iteration {i+1}/{self.iterations}, Loss: {loss:.4f}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练逻辑回归模型

        Args:
            X (np.ndarray): 特征矩阵
            y (np.ndarray): 标签向量
        """
        self.gradient_descent(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本属于正类的概率

        Args:
            X (np.ndarray): 特征矩阵

        Returns:
            np.ndarray: 预测概率
        """
        model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本的标签

        Args:
            X (np.ndarray): 特征矩阵

        Returns:
            np.ndarray: 预测标签
        """
        return self.predict_proba(X) >= 0.5

def main():
    # 示例数据
    X_train = np.array([[0.2, 0.8], [0.5, 0.5], [0.9, 0.1], [0.4, 0.6], [0.7, 0.3]])
    y_train = np.array([0, 0, 1, 0, 1])

    # 初始化逻辑回归模型
    lr_model = LogisticRegressionModel(learning_rate=0.01, iterations=1000)

    # 训练模型
    lr_model.fit(X_train, y_train)

    # 预测
    X_test = np.array([[0.3, 0.7], [0.8, 0.2]])
    predictions = lr_model.predict(X_test)

    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
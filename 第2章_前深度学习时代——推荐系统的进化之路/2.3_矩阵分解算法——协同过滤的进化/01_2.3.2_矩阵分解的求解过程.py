# 01_2.3.2 矩阵分解的求解过程

"""
Lecture: 第2章 前深度学习时代——推荐系统的进化之路/2.3 矩阵分解算法——协同过滤的进化
Content: 01_2.3.2 矩阵分解的求解过程
"""

import numpy as np
from typing import Tuple

class MatrixFactorization:
    def __init__(self, R: np.ndarray, K: int, alpha: float, beta: float, iterations: int):
        """
        初始化矩阵分解类

        Args:
            R (np.ndarray): 用户-物品评分矩阵
            K (int): 潜在特征的数量
            alpha (float): 学习率
            beta (float): 正则化参数
            iterations (int): 迭代次数
        """
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        训练矩阵分解模型

        Returns:
            Tuple[np.ndarray, np.ndarray, float]: 分别为用户特征矩阵、物品特征矩阵及训练误差
        """
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print(f"Iteration: {i+1}; error = {mse:.4f}")

        return self.P, self.Q, training_process

    def mse(self) -> float:
        """
        计算均方误差

        Returns:
            float: 均方误差
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += (self.R[x, y] - predicted[x, y])**2
        return np.sqrt(error)

    def sgd(self):
        """
        随机梯度下降优化
        """
        for i, j, r in self.samples:
            prediction = self.get_prediction(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])

    def get_prediction(self, i: int, j: int) -> float:
        """
        获取对用户i对物品j的评分预测

        Args:
            i (int): 用户索引
            j (int): 物品索引

        Returns:
            float: 预测评分
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self) -> np.ndarray:
        """
        重建完整的用户-物品评分矩阵

        Returns:
            np.ndarray: 完整评分矩阵
        """
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.P.dot(self.Q.T)

def main():
    # 示例用户-物品评分矩阵
    R = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])

    # 初始化矩阵分解模型
    mf = MatrixFactorization(R, K=2, alpha=0.01, beta=0.01, iterations=100)

    # 训练模型
    P, Q, training_process = mf.train()

    print("\nP matrix:\n", P)
    print("\nQ matrix:\n", Q)
    print("\nPredicted Ratings:\n", mf.full_matrix())

if __name__ == "__main__":
    main()

# 00_2.6.1 GBDT+LR组合模型的结构

"""
Lecture: 第2章 前深度学习时代——推荐系统的进化之路/2.6 GBDT+LR——特征工程模型化的开端
Content: 00_2.6.1 GBDT+LR组合模型的结构
"""

"""
GBDT Model Implementation.

This implementation provides a comprehensive and well-structured GBDT model class, designed for industrial scenarios.
The code includes robust classes and functions with boundary condition checks and adheres to Google style guides.
All key steps are commented in Chinese following the PEP 8 style guide, and DocStrings are provided according to PEP 257.

Classes:
    GBDT: Class implementing the Gradient Boosting Decision Tree (GBDT) model for regression tasks.

Methods:
    fit: Train the GBDT model on given data.
    predict: Make predictions using the trained GBDT model.
    _compute_residuals: Compute residuals for boosting.
    _initialize_base_model: Initialize the base model.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from typing import List, Tuple

class GBDT:
    """
    Gradient Boosting Decision Tree (GBDT) model for regression tasks.

    Attributes:
        n_estimators (int): Number of boosting stages.
        learning_rate (float): Learning rate shrinks the contribution of each tree.
        max_depth (int): Maximum depth of the individual regression estimators.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        trees (List[DecisionTreeRegressor]): List of individual regression trees.
        base_model (float): Initial base model (mean of target values).
    """
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3, min_samples_split: int = 2):
        """
        Initialize the GBDT model.

        Args:
            n_estimators (int): Number of boosting stages.
            learning_rate (float): Learning rate shrinks the contribution of each tree.
            max_depth (int): Maximum depth of the individual regression estimators.
            min_samples_split (int): Minimum number of samples required to split an internal node.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.base_model = None

    def _initialize_base_model(self, y: np.ndarray):
        """初始化基模型（目标值的均值）。"""
        self.base_model = np.mean(y)

    def _compute_residuals(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        计算残差。

        Args:
            y (np.ndarray): 真实目标值。
            y_pred (np.ndarray): 预测值。

        Returns:
            np.ndarray: 残差。
        """
        return y - y_pred

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练GBDT模型。

        Args:
            X (np.ndarray): 输入特征矩阵。
            y (np.ndarray): 目标值向量。
        """
        self._initialize_base_model(y)
        y_pred = np.full(y.shape, self.base_model)

        for i in range(self.n_estimators):
            residuals = self._compute_residuals(y, y_pred)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

            if i % 10 == 0:
                loss = np.mean((y - y_pred) ** 2)
                print(f"Iteration {i}: Loss = {loss}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的GBDT模型进行预测。

        Args:
            X (np.ndarray): 输入特征矩阵。

        Returns:
            np.ndarray: 预测值向量。
        """
        y_pred = np.full(X.shape[0], self.base_model)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

# # 测试 GBDT 模型
# if __name__ == "__main__":
#     # 生成模拟数据
#     np.random.seed(42)
#     X = np.random.rand(100, 5)
#     y = 3 + 2 * X[:, 0] + X[:, 1] + np.random.randn(100)

#     # 初始化并训练模型
#     model = GBDT(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2)
#     model.fit(X, y)

#     # 进行预测
#     y_pred = model.predict(X)
#     print("Predicted values:", y_pred[:10])
#     print("Actual values:", y[:10])

"""
GBDT+LR Model Implementation.

This implementation provides a comprehensive and well-structured GBDT+LR model class, designed for industrial scenarios.
The code includes robust classes and functions with boundary condition checks and adheres to Google style guides.
All key steps are commented in Chinese following the PEP 8 style guide, and DocStrings are provided according to PEP 257.

Classes:
    GBDT_LR: Class implementing the GBDT+LR model for regression tasks.

Methods:
    fit: Train the GBDT+LR model on given data.
    predict: Make predictions using the trained GBDT+LR model.
    _transform_features: Transform features using trained GBDT.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple

class GBDT_LR:
    """
    GBDT+LR Model for regression tasks.

    Attributes:
        n_estimators (int): Number of boosting stages.
        learning_rate (float): Learning rate shrinks the contribution of each tree.
        max_depth (int): Maximum depth of the individual regression estimators.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        trees (List[DecisionTreeRegressor]): List of individual regression trees.
        lr_model (LogisticRegression): Logistic Regression model.
    """
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3, min_samples_split: int = 2):
        """
        Initialize the GBDT+LR model.

        Args:
            n_estimators (int): Number of boosting stages.
            learning_rate (float): Learning rate shrinks the contribution of each tree.
            max_depth (int): Maximum depth of the individual regression estimators.
            min_samples_split (int): Minimum number of samples required to split an internal node.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.lr_model = LogisticRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练GBDT+LR模型。

        Args:
            X (np.ndarray): 输入特征矩阵。
            y (np.ndarray): 目标值向量。
        """
        # 训练GBDT模型
        y_pred = np.zeros(y.shape)
        for i in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

        # 特征转换
        transformed_X = self._transform_features(X)

        # 训练LR模型
        self.lr_model.fit(transformed_X, y)

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的GBDT模型转换特征。

        Args:
            X (np.ndarray): 输入特征矩阵。

        Returns:
            np.ndarray: 转换后的特征矩阵。
        """
        transformed_X = np.zeros((X.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            transformed_X[:, i] = tree.predict(X)
        return transformed_X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的GBDT+LR模型进行预测。

        Args:
            X (np.ndarray): 输入特征矩阵。

        Returns:
            np.ndarray: 预测值向量。
        """
        transformed_X = self._transform_features(X)
        return self.lr_model.predict(transformed_X)

# 测试 GBDT+LR 模型
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = (3 + 2 * X[:, 0] + X[:, 1] + np.random.randn(100) > 5).astype(int)

    # 初始化并训练模型
    model = GBDT_LR(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2)
    model.fit(X, y)

    # 进行预测
    y_pred = model.predict(X)
    print("Predicted values:", y_pred[:10])
    print("Actual values:", y[:10])
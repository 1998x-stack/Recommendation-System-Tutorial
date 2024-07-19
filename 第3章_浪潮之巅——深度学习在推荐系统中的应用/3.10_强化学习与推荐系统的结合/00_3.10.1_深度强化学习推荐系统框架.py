# 00_3.10.1 深度强化学习推荐系统框架

"""
Lecture: 第3章 浪潮之巅——深度学习在推荐系统中的应用/3.10 强化学习与推荐系统的结合
Content: 00_3.10.1 深度强化学习推荐系统框架
"""

import numpy as np
import random
from typing import List, Tuple

class NewsRecommendationEnv:
    def __init__(self, user_features: np.ndarray, news_features: np.ndarray):
        """
        新闻推荐环境类
        
        Args:
            user_features (np.ndarray): 用户特征矩阵
            news_features (np.ndarray): 新闻特征矩阵
        """
        self.user_features = user_features
        self.news_features = news_features
        self.num_users = user_features.shape[0]
        self.num_news = news_features.shape[0]
        self.current_user_index = 0
    
    def reset(self) -> np.ndarray:
        """
        重置环境，随机选择一个用户
        
        Returns:
            np.ndarray: 当前用户特征
        """
        self.current_user_index = random.randint(0, self.num_users - 1)
        current_user_features = self.user_features[self.current_user_index]
        return current_user_features
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        执行动作，模拟用户对推荐内容的反馈
        
        Args:
            action (int): 推荐的新闻索引
        
        Returns:
            Tuple[np.ndarray, float, bool]: 下一个状态，奖励，是否结束
        """
        user = self.user_features[self.current_user_index]
        news = self.news_features[action]
        state = np.concatenate((user, news))
        
        # 模拟用户反馈
        clicked = self._user_click_simulation(user, news)
        reward = 1.0 if clicked else 0.0
        done = bool(random.choice([True, False]))  # 随机结束
        next_state = self.reset() if done else state
        
        return next_state, reward, done
    
    def _user_click_simulation(self, user: np.ndarray, news: np.ndarray) -> bool:
        """
        模拟用户点击行为
        
        Args:
            user (np.ndarray): 用户特征向量
            news (np.ndarray): 新闻特征向量
        
        Returns:
            bool: 是否点击
        """
        probability = np.dot(user, news) / (np.linalg.norm(user) * np.linalg.norm(news))
        return random.random() < probability

# 示例用户和新闻特征
user_features = np.random.rand(100, 5)  # 100个用户，每个用户5个特征
news_features = np.random.rand(50, 5)   # 50篇新闻，每篇新闻5个特征

env = NewsRecommendationEnv(user_features, news_features)

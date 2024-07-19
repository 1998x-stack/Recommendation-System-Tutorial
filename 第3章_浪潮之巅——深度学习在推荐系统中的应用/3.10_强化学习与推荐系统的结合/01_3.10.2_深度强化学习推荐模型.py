# 01_3.10.2 深度强化学习推荐模型

"""
Lecture: 第3章 浪潮之巅——深度学习在推荐系统中的应用/3.10 强化学习与推荐系统的结合
Content: 01_3.10.2 深度强化学习推荐模型
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
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Tuple

# 深度Q网络模型定义
class DQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(128, 128)  # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(128, action_size)  # 第二个隐藏层到输出层
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))  # 通过第一个隐藏层并使用ReLU激活函数
        x = torch.relu(self.fc2(x))  # 通过第二个隐藏层并使用ReLU激活函数
        x = self.fc3(x)  # 通过输出层
        return x

# 深度强化学习推荐模型类定义
class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # 经验回放缓冲区
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减系数
        self.learning_rate = 0.001  # 学习率
        self.model = DQNetwork(state_size, action_size)  # 评估模型
        self.target_model = DQNetwork(state_size, action_size)  # 目标模型
        self.update_target_model()  # 初始化目标模型权重
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # 优化器
        self.criterion = nn.MSELoss()  # 损失函数

    def update_target_model(self):
        """将目标模型的权重更新为评估模型的权重"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """将经验存储到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """
        根据当前策略选择动作
        如果随机值小于探索率，则随机选择一个动作
        否则选择模型预测值最大的动作
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size: int):
        """从经验回放缓冲区中采样并训练模型"""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = self.model(state).clone().detach()  # 克隆当前状态下的Q值
            if done:
                target[action] = reward  # 如果结束，目标Q值为奖励
            else:
                t = self.target_model(next_state).detach()
                target[action] = reward + self.gamma * torch.max(t)  # 更新目标Q值
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name: str):
        """加载模型权重"""
        self.model.load_state_dict(torch.load(name))

    def save(self, name: str):
        """保存模型权重"""
        torch.save(self.model.state_dict(), name)

# 数据处理和环境模拟
def preprocess_data(user_features: np.ndarray, context_features: np.ndarray) -> np.ndarray:
    """数据预处理，生成状态向量"""
    state = np.concatenate((user_features, context_features))
    return state

def simulate_environment(state: np.ndarray, action: int) -> Tuple[np.ndarray, float, bool]:
    """模拟环境反馈"""
    next_state = state + np.random.randn(len(state)) * 0.1  # 随机噪声模拟状态转移
    reward = random.choice([1, 0])  # 随机奖励
    done = random.choice([True, False])  # 随机终止
    return next_state, reward, done

# 示例使用
if __name__ == "__main__":
    EPISODES = 1000  # 总训练回合数
    state_size = 8  # 假设状态维度为8（用户特征+环境特征）
    action_size = 4  # 假设动作维度为4
    batch_size = 32  # 批量大小

    agent = DQNAgent(state_size, action_size)
    cumulative_rewards = []  # 存储每回合的累积奖励

    for e in range(EPISODES):
        user_features = np.random.rand(4)  # 随机生成用户特征
        context_features = np.random.rand(4)  # 随机生成环境特征
        state = preprocess_data(user_features, context_features)
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = simulate_environment(state, action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                agent.update_target_model()
                print(f"episode: {e}/{EPISODES}, score: {time}, total reward: {total_reward}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        cumulative_rewards.append(total_reward)
        if e % 10 == 0:
            agent.save(f"dqn_model_{e}.pth")

    # 绘制累积奖励图
    plt.plot(cumulative_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards over Episodes')
    plt.show()

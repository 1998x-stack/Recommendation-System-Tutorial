# 02_3.10.3 DRN的学习过程

"""
Lecture: 第3章 浪潮之巅——深度学习在推荐系统中的应用/3.10 强化学习与推荐系统的结合
Content: 02_3.10.3 DRN的学习过程
"""

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Tuple

# 定义深度Q网络模型
class DQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义深度强化学习推荐模型类
class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQNetwork(state_size, action_size)
        self.target_model = DQNetwork(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size: int):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = self.model(state).clone().detach()
            if done:
                target[action] = reward
            else:
                t = self.target_model(next_state).detach()
                target[action] = reward + self.gamma * torch.max(t)
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name: str):
        self.model.load_state_dict(torch.load(name))

    def save(self, name: str):
        torch.save(self.model.state_dict(), name)

# 数据处理和环境模拟
def preprocess_data(user_features: np.ndarray, context_features: np.ndarray) -> np.ndarray:
    state = np.concatenate((user_features, context_features))
    return state

def simulate_environment(state: np.ndarray, action: int) -> Tuple[np.ndarray, float, bool]:
    next_state = state + np.random.randn(len(state)) * 0.1
    reward = random.choice([1, 0])
    done = random.choice([True, False])
    return next_state, reward, done

# 竞争梯度下降算法
def dueling_bandit_gradient_descent(agent: DQNAgent, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
    current_params = [param.clone() for param in agent.model.parameters()]
    perturbation = [torch.randn_like(param) * 0.1 for param in agent.model.parameters()]
    new_params = [param + delta for param, delta in zip(current_params, perturbation)]
    
    for param, new_param in zip(agent.model.parameters(), new_params):
        param.data.copy_(new_param.data)
    
    original_state_action_values = agent.model(torch.FloatTensor(state))
    perturbed_state_action_values = agent.model(torch.FloatTensor(state))
    
    original_reward = original_state_action_values[action].item()
    perturbed_reward = perturbed_state_action_values[action].item()
    
    if perturbed_reward > original_reward:
        agent.remember(state, action, reward, next_state, done)
        agent.replay(32)
    else:
        for param, original_param in zip(agent.model.parameters(), current_params):
            param.data.copy_(original_param.data)

# 示例使用
if __name__ == "__main__":
    EPISODES = 1000
    state_size = 8
    action_size = 4
    batch_size = 32

    agent = DQNAgent(state_size, action_size)
    cumulative_rewards = []

    for e in range(EPISODES):
        user_features = np.random.rand(4)
        context_features = np.random.rand(4)
        state = preprocess_data(user_features, context_features)
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = simulate_environment(state, action)
            next_state = np.reshape(next_state, [1, state_size])
            dueling_bandit_gradient_descent(agent, state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                agent.update_target_model()
                print(f"episode: {e}/{EPISODES}, score: {time}, total reward: {total_reward}, e: {agent.epsilon:.2}")
                break
        cumulative_rewards.append(total_reward)
        if e % 10 == 0:
            agent.save(f"dqn_model_{e}.pth")

    plt.plot(cumulative_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards over Episodes')
    plt.show()

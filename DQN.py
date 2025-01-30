
import gymnasium_robotics
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
import matplotlib.pyplot as plt

env = gym.make("FetchReachDense-v4", render_mode=None)
obs_dim_robot = env.observation_space["observation"].shape[0]
obs_dim_goal = env.observation_space["desired_goal"].shape[0]
observation_dim = obs_dim_robot + obs_dim_goal
action_dim = env.action_space.shape[0]

gamma = 0.95
learning_rate = 1e-4
batch_size = 128
buffer_size = 10000
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.995
target_update_frequency = 5
episodes = 6000

n_discrete_actions = 5
action_space = np.linspace(-1, 1, n_discrete_actions)
discrete_actions = np.array(
    np.meshgrid(*[action_space] * action_dim)
).T.reshape(-1, action_dim)

class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

q_network = QNetwork(observation_dim, len(discrete_actions))
target_network = QNetwork(observation_dim, len(discrete_actions))
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
replay_buffer = deque(maxlen=buffer_size)

def select_action(state, epsilon):
    if random.random() < epsilon:
        action_idx = random.randint(0, len(discrete_actions) - 1)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = q_network(state_tensor)
        action_idx = q_values.argmax(dim=1).item()
    return action_idx, discrete_actions[action_idx]

def store_experience(buffer, experience):
    buffer.append(experience)

def sample_batch(buffer, batch_size):
    return random.sample(buffer, batch_size)

def update_network(batch):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    with torch.no_grad():
        next_q_values = target_network(next_states).max(dim=1)[0]
        targets = rewards + (1 - dones) * gamma * next_q_values
    current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
    loss = nn.MSELoss()(current_q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

epsilon = epsilon_start
rewards_list = []

for episode in range(episodes):
    if episode == 1200:
        env = gym.make("FetchReachDense-v4", render_mode="human")
    obs_dict, info = env.reset()
    state = np.concatenate([obs_dict["observation"], obs_dict["desired_goal"]])
    total_reward = 0

    for t in range(1000):
        if episode >= 1200:
            env.render()
        action_idx, action = select_action(state, epsilon)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        next_obs_dict, reward, terminated, truncated, info = env.step(action)
        next_state = np.concatenate([next_obs_dict["observation"], next_obs_dict["desired_goal"]])
        store_experience(replay_buffer, (state, action_idx, reward, next_state, terminated))
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break
        if len(replay_buffer) >= batch_size:
            batch = sample_batch(replay_buffer, batch_size)
            update_network(batch)

    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    if episode % target_update_frequency == 0:
        target_network.load_state_dict(q_network.state_dict())
    rewards_list.append(total_reward)
    print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

env.close()

average_rewards = []
std_rewards = []
window_size = 50

for i in range(0, len(rewards_list), window_size):
    window_rewards = rewards_list[i:i + window_size]
    avg_reward = sum(window_rewards) / len(window_rewards)
    std_reward = np.std(window_rewards)
    average_rewards.append(avg_reward)
    std_rewards.append(std_reward)

plt.figure(figsize=(12, 6))
plt.plot(average_rewards, label='Average Reward (per 10 episodes)', color='blue')
plt.fill_between(
    range(len(average_rewards)),
    np.array(average_rewards) - np.array(std_rewards),
    np.array(average_rewards) + np.array(std_rewards),
    color='blue',
    alpha=0.2,
    label='Standard Deviation'
)
plt.xlabel('Episode (x10)')
plt.ylabel('Average Total Reward')
plt.title('Average Total Rewards Over Episodes with Standard Deviation')
plt.legend()
plt.grid(True)
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque

# Neural Network for the Critic (Q-function)
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Neural Network for the Actor (Policy)
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, action_size)
        self.fc_log_std = nn.Linear(hidden_size, action_size)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

# Replay Buffer for Experience Replay
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self):
        experiences = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.float32),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)

# Soft Actor-Critic (SAC) Agent
class SACAgent:
    def __init__(self, state_size, action_size, hidden_size=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, buffer_size=1000000, batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        self.actor = Actor(state_size, action_size, hidden_size)
        self.q1 = QNetwork(state_size, action_size, hidden_size)
        self.q2 = QNetwork(state_size, action_size, hidden_size)
        self.q1_target = QNetwork(state_size, action_size, hidden_size)
        self.q2_target = QNetwork(state_size, action_size, hidden_size)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size, batch_size)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target_next = self.q1_target(next_states, next_actions)
            q2_target_next = self.q2_target(next_states, next_actions)
            q_target_next = torch.min(q1_target_next, q2_target_next)
            q_targets = rewards + (1 - dones) * self.gamma * (q_target_next - self.alpha * next_log_probs)

        q1_expected = self.q1(states, actions)
        q2_expected = self.q2(states, actions)

        q1_loss = F.mse_loss(q1_expected, q_targets)
        q2_loss = F.mse_loss(q2_expected, q_targets)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        actions_pred, log_probs = self.actor.sample(states)
        q1_pred = self.q1(states, actions_pred)
        q2_pred = self.q2(states, actions_pred)
        q_pred = torch.min(q1_pred, q2_pred)

        actor_loss = (self.alpha * log_probs - q_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

# Training the SAC agent in a Gym environment
env = gym.make('Pendulum-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
agent = SACAgent(state_size, action_size)

episodes = 1000
max_t = 1000

for e in range(episodes):
    state = env.reset()
    total_reward = 0
    for t in range(max_t):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

env.close()
""""
Required Libraries: pip install torch gym numpy
Explanation:
    Q-Network (QNetwork):
        This class implements the critic network, which approximates the Q-value function. The network takes both the state and action as input and outputs the Q-value. It is structured with two hidden layers followed by a linear output layer.
    Actor Network (Actor):
        The actor network is responsible for modeling the policy. It outputs the mean and log standard deviation of the action distribution. The sample method returns an action by sampling from this distribution and applying the tanh function to ensure the action remains within valid bounds.
    Replay Buffer (ReplayBuffer):
        The replay buffer stores past experiences (state, action, reward, next state, done) and samples random minibatches for training. This ensures that the agent learns from diverse experiences and mitigates correlation in sequential data.
    Soft Actor-Critic Agent (SACAgent):
        Actor and Critic: The agent contains two critic networks (Q1 and Q2) and an actor network. It also maintains target networks for the critics, which are softly updated to improve stability during training.
        Learning: The learn method uses the Bellman equation to calculate target Q-values. The actor loss is calculated based on the expected Q-values and the entropy of the action distribution (controlled by the alpha parameter), promoting exploration.
        Soft Update: Target networks are softly updated by mixing their parameters with those of the corresponding local networks.
    Training Loop:
        The agent is trained in the Pendulum-v1 environment over a specified number of episodes. During each episode, the agent interacts with the environment and learns from the collected experiences. The total reward for each episode is printed as a### Explanation (continued)
    Training Loop (continued):
        The agent is trained in the Pendulum-v1 environment over a specified number of episodes. During each episode, the agent interacts with the environment and learns from the collected experiences. The total reward for each episode is printed as a metric for monitoring the agent's performance.
"""
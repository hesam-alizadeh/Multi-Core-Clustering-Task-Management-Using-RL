import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from collections import deque
import random

# Neural Network for the Q-function
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Memory
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

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

# Double DQN Agent
class DDQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lr=0.001, gamma=0.99, tau=0.001, buffer_size=10000, batch_size=64, update_every=4):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every

        self.qnetwork_local = QNetwork(state_size, action_size, hidden_size)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, epsilon=0.0):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Double DQN: using local network for action selection
        next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network with soft update
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_param.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

# Training the agent in a Gym environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DDQNAgent(state_size, action_size)

episodes = 1000
max_t = 1000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

for e in range(episodes):
    state = env.reset()
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** e))
    for t in range(max_t):
        action = agent.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    print(f"Episode {e+1}/{episodes}, Score: {t+1}")

env.close()


"""
Libraries Required: pip install numpy torch gym
Explanation:
    Neural Network (QNetwork): The QNetwork class defines a simple fully connected neural network with two hidden layers. This network takes the state as input and outputs Q-values for each possible action.
    Experience Replay (ReplayBuffer): The ReplayBuffer class stores the agent's experiences and allows the agent to learn from randomly sampled mini-batches. This helps in breaking the correlation between consecutive experiences, stabilizing the training process.
    Double DQN Agent (DDQNAgent):
        Action Selection: The act method selects actions based on an epsilon-greedy policy.
        Learning (learn): During learning, the agent computes the target Q-value using the Double DQN approach: the next action is selected using the local network, and the target Q-value is obtained from the target network.
        Soft Update: The target network is updated slowly using a soft update strategy, where the target network's parameters are slowly moved towards the local network's parameters.
    Training Loop:
"""
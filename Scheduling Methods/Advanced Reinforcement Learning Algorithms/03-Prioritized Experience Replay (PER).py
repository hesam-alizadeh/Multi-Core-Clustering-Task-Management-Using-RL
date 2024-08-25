import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from collections import deque
import random
import math

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

# Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, batch_size, alpha=0.6):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.alpha = alpha
        self.priorities = deque(maxlen=buffer_size)
        self.epsilon = 1e-6

    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.buffer else 1.0
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.priorities.append(max_priority)

    def sample(self, beta=0.4):
        if len(self.buffer) == len(self.priorities):
            priorities = np.array(self.priorities, dtype=np.float32)
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()

            indices = np.random.choice(len(self.buffer), self.batch_size, p=probabilities)
            experiences = [self.buffer[idx] for idx in indices]
            weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
            weights /= weights.max()
            weights = torch.tensor(weights, dtype=torch.float32)

            states, actions, rewards, next_states, dones = zip(*experiences)
            return (torch.tensor(states, dtype=torch.float32),
                    torch.tensor(actions, dtype=torch.int64),
                    torch.tensor(rewards, dtype=torch.float32),
                    torch.tensor(next_states, dtype=torch.float32),
                    torch.tensor(dones, dtype=torch.float32),
                    indices,
                    weights)

    def update_priorities(self, indices, errors):
        errors += self.epsilon
        clipped_errors = np.minimum(errors, 1.0)
        for idx, error in zip(indices, clipped_errors):
            self.priorities[idx] = error

    def __len__(self):
        return len(self.buffer)

# Double DQN Agent with Prioritized Experience Replay
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
        self.memory = PrioritizedReplayBuffer(buffer_size, batch_size)
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
        states, actions, rewards, next_states, dones, indices, weights = experiences

        next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))
        errors = torch.abs(Q_expected - Q_targets).detach().cpu().numpy()
        self.memory.update_priorities(indices, errors)

        loss = (weights * F.mse_loss(Q_expected, Q_targets, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
""""
Required Libraries: pip install numpy torch gym
Explanation:
    Neural Network (QNetwork):
        The QNetwork class defines a fully connected neural network that serves as the Q-function approximator. The network consists of two hidden layers with ReLU activation functions, followed by an output layer corresponding to the Q-values for each action.
    Prioritized Replay Buffer (PrioritizedReplayBuffer):
        The PrioritizedReplayBuffer stores agent experiences and assigns a priority to each experience based on the magnitude of its Temporal Difference (TD) error. When sampling experiences for training, experiences with higher priorities are more likely to be chosen, improving learning efficiency by focusing on more informative experiences.
        Sample Method: The method uses a stochastic approach to select experiences based on their priorities. The experiences are then returned along with importance-sampling weights to adjust for the bias introduced by prioritized sampling.
        Update Priorities Method: After learning from a batch of experiences, the priorities are updated based on the TD errors to reflect the current learning needs.
    Double DQN Agent with PER (DDQNAgent):
        Action Selection: The act method follows an epsilon-greedy strategy to select actions.
        Learning (learn): The agent learns by minimizing the weighted TD error using the experiences sampled from the prioritized replay buffer. The Double DQN approach is employed to reduce overestimation bias by using the target network for action evaluation.
        Soft Update: The target network is slowly updated towards the local network's parameters using a soft update strategy.
    Training Loop:
        The agent is trained over a number of episodes in the CartPole environment. The epsilon parameter is gradually reduced, lowering the exploration rate as the agent becomes more confident in its learned policy.
"""
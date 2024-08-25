import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Hyperparameters
gamma = 0.99          # Discount factor for future rewards
epsilon = 1.0         # Initial exploration rate
epsilon_min = 0.1     # Minimum exploration rate
epsilon_decay = 0.995 # Epsilon decay rate
learning_rate = 0.001 # Learning rate for the optimizer
batch_size = 64       # Batch size for experience replay
memory_size = 100000  # Size of the experience replay buffer
target_update_freq = 10 # Frequency of target network updates

# Environment
env = gym.make('CartPole-v1')
state_shape = env.observation_space.shape
num_actions = env.action_space.n

# Q-Network
def build_q_network():
    model = tf.keras.Sequential([
        layers.Dense(24, activation='relu', input_shape=state_shape),
        layers.Dense(24, activation='relu'),
        layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = []
        self.size = size

    def add(self, experience):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

# DQN Agent
class DQNAgent:
    def __init__(self):
        self.q_network = build_q_network()
        self.target_q_network = build_q_network()
        self.target_q_network.set_weights(self.q_network.get_weights())
        self.memory = ReplayBuffer(memory_size)
        self.epsilon = epsilon

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(num_actions)
        q_values = self.q_network.predict(state[np.newaxis])
        return np.argmax(q_values[0])

    def train(self):
        if len(self.memory.buffer) < batch_size:
            return

        minibatch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)
        actions = np.array(actions)

        q_values_next = self.target_q_network.predict(next_states)
        target_q_values = self.q_network.predict(states)

        for i in range(batch_size):
            target_q_values[i][actions[i]] = rewards[i] + (1 - dones[i]) * gamma * np.amax(q_values_next[i])

        self.q_network.train_on_batch(states, target_q_values)

    def update_epsilon(self):
        self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)

    def update_target_network(self):
        self.target_q_network.set_weights(self.q_network.get_weights())

# Training Loop
agent = DQNAgent()
episodes = 1000
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.memory.add((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        agent.train()

    agent.update_epsilon()

    if episode % target_update_freq == 0:
        agent.update_target_network()

    print(f"Episode: {episode+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

env.close()


"""
Required Libraries: pip install tensorflow numpy gym
Explanation of the Code:
This code implements the Deep Q-Network (DQN) algorithm, a reinforcement learning technique used for training agents in environments modeled as Markov Decision Processes (MDPs).
    Q-Network: The Q-network is a deep neural network that approximates the Q-values for each action given a state. The Q-values represent the expected future rewards of taking a certain action in a given state.
    Experience Replay: The agent uses a replay buffer to store experiences (state, action, reward, next state, done) and samples random minibatches from this buffer during training. This technique helps to break the correlation between consecutive experiences, which improves the stability of the learning process.
    Target Network: A separate target network is used to compute the Q-value targets during training. The weights of the target network are periodically updated to match the weights of the main Q-network, which helps to stabilize training by reducing oscillations.
    Epsilon-Greedy Policy: The agent follows an epsilon-greedy policy where it chooses random actions with probability epsilon and selects the action with the highest Q-value with probability 1-epsilon. The epsilon value decreases over time, gradually shifting the policy from exploration to exploitation.
"""
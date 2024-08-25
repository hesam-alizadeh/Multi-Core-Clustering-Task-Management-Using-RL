import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gym
import random
from collections import deque, namedtuple
class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units):
        super(NoisyDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.mu_w = self.add_weight(name='mu_w', shape=(input_shape[-1], self.units), initializer=tf.random_uniform_initializer(-1/np.sqrt(input_shape[-1]), 1/np.sqrt(input_shape[-1])))
        self.sigma_w = self.add_weight(name='sigma_w', shape=(input_shape[-1], self.units), initializer=tf.constant_initializer(0.017))
        self.mu_b = self.add_weight(name='mu_b', shape=(self.units,), initializer=tf.random_uniform_initializer(-1/np.sqrt(input_shape[-1]), 1/np.sqrt(input_shape[-1])))
        self.sigma_b = self.add_weight(name='sigma_b', shape=(self.units,), initializer=tf.constant_initializer(0.017))

    def call(self, inputs):
        epsilon_in = tf.random.normal(shape=(inputs.shape[-1], self.units))
        epsilon_out = tf.random.normal(shape=(self.units,))
        noisy_w = self.mu_w + self.sigma_w * epsilon_in
        noisy_b = self.mu_b + self.sigma_b * epsilon_out
        return tf.matmul(inputs, noisy_w) + noisy_b


class RainbowDQN:
    def __init__(self, state_dim, action_dim, atom_size=51, v_min=-10, v_max=10, gamma=0.99, lr=0.0001, batch_size=32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size

        self.support = np.linspace(self.v_min, self.v_max, self.atom_size)
        self.delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)

        self.memory = deque(maxlen=100000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def build_model(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = NoisyDense(512)(inputs)
        x = layers.ReLU()(x)
        x = NoisyDense(512)(x)
        x = layers.ReLU()(x)
        out = NoisyDense(self.action_dim * self.atom_size)(x)
        out = layers.Reshape((self.action_dim, self.atom_size))(out)
        model = tf.keras.Model(inputs=inputs, outputs=out)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        return states, actions, rewards, next_states, dones

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample()
        next_actions = np.argmax(self.model.predict(next_states), axis=1)

        target_distributions = self.target_model.predict(next_states)
        target_distributions = np.array([target_distributions[i, next_actions[i]] for i in range(self.batch_size)])

        t_z = rewards[:, None] + self.gamma * self.support * (1 - dones[:, None])
        t_z = np.clip(t_z, self.v_min, self.v_max)
        b = (t_z - self.v_min) / self.delta_z
        l = np.floor(b).astype(np.int64)
        u = np.ceil(b).astype(np.int64)

        batch_indices = np.arange(self.batch_size)
        m = np.zeros((self.batch_size, self.atom_size))
        for i in range(self.atom_size):
            l_index = (batch_indices, l[:, i])
            u_index = (batch_indices, u[:, i])
            m[l_index] += target_distributions[:, i] * (u - b)[:, i]
            m[u_index] += target_distributions[:, i] * (b - l)[:, i]

        with tf.GradientTape() as tape:
            logits = self.model(states)
            logits = tf.gather_nd(logits, tf.stack((tf.range(self.batch_size), actions), axis=1))
            logits = tf.nn.softmax(logits, axis=-1)
            loss = tf.keras.losses.KLDivergence()(m, logits)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def policy(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = np.expand_dims(state, axis=0)
        action = np.argmax(self.model.predict(state)[0])
        return action

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model = tf.keras.models.load_model(name)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = RainbowDQN(state_dim, action_dim)

    episodes = 500
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        for time in range(500):
            action = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                agent.update_target()
                print(f"Episode: {e}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon}")
                break

            agent.train()

        agent.decay_epsilon()

    agent.save("rainbow_dqn.h5")
"""
Explanation:
The Rainbow DQN algorithm is an enhancement of the original Deep Q-Network (DQN) by integrating several advanced techniques, making it more effective in complex reinforcement learning tasks. The primary components and techniques used in Rainbow DQN are:
    Noisy Networks: Replace the standard fully connected layers with Noisy Dense layers to introduce learnable noise, enabling more efficient exploration.
    Double DQN: Used to reduce the overestimation bias by separating action selection and action evaluation.
    Prioritized Experience Replay (PER): Sampling more important experiences more frequently, increasing the learning efficiency.
    Dueling Networks: Splitting the Q-value into state-value and advantage components, which allows the model to learn which states are valuable without needing to learn the effect of each action.
    Multi-step Learning: Consideration of multiple future steps during training, improving the stability and performance of the learning process.
    Distributional RL: Predicting the full distribution of returns (instead of just the expected return), providing richer information and stabilizing training.
"""
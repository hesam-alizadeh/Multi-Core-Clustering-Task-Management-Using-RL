import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gym
from collections import deque
import random
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_deviation = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_deviation * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, state_dim))
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_dim))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    def sample(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        return state_batch, action_batch, reward_batch, next_state_batch


class DDPG:
    def __init__(self):
        self.gamma = 0.99
        self.tau = 0.005
        self.buffer = Buffer()

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.critic_optimizer = tf.keras.optimizers.Adam(0.002)
        self.actor_optimizer = tf.keras.optimizers.Adam(0.001)

        self.noise = OUActionNoise(mean=np.zeros(action_dim), std_deviation=float(0.2) * np.ones(action_dim))

    def get_actor(self):
        inputs = layers.Input(shape=(state_dim,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(action_dim, activation="tanh")(out)
        outputs = outputs * upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic(self):
        state_input = layers.Input(shape=(state_dim))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        action_input = layers.Input(shape=(action_dim))
        action_out = layers.Dense(32, activation="relu")(action_input)

        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        model = tf.keras.Model([state_input, action_input], outputs)
        return model

    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = self.noise()
        sampled_actions = sampled_actions.numpy() + noise
        legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
        return [np.squeeze(legal_action)]

    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def learn(self):
        state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample()

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch)
            y = reward_batch + self.gamma * self.target_critic([next_state_batch, target_actions])
            critic_value = self.critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch)
            critic_value = self.critic_model([state_batch, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

    def train(self, env, max_episodes):
        for ep in range(max_episodes):
            prev_state = env.reset()
            episodic_reward = 0

            while True:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action = self.policy(tf_prev_state)

                state, reward, done, _ = env.step(action)
                self.buffer.record((prev_state, action, reward, state))
                episodic_reward += reward

                self.learn()
                self.update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
                self.update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

                if done:
                    break

                prev_state = state

            print(f"Episode: {ep + 1}, Reward: {episodic_reward}")


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    agent = DDPG()
    agent.train(env, max_episodes=100)
"""
Explanation:
The Deep Deterministic Policy Gradient (DDPG) is an advanced model-free, off-policy actor-critic algorithm used for continuous action spaces. Unlike traditional policy gradient methods, DDPG is deterministic, meaning it directly outputs an action rather than a probability distribution over actions. Here are the key components:
    Actor Network: The actor network approximates the optimal policy, producing continuous actions based on the current state.
    Critic Network: The critic network evaluates the quality of the chosen actions by estimating the expected return (or Q-value) from a given state-action pair.
    Target Networks: These are slowly-updated versions of the actor and critic networks, which help stabilize training by reducing the correlation between actions and target values.
    Ornstein-Uhlenbeck (OU) Noise: This noise is added to the action outputs during training to encourage exploration, especially in environments with inertia, such as physical control tasks.
    Experience Replay: The agent stores past experiences in a replay buffer and samples from it to break the temporal correlation between consecutive experiences. This technique improves the stability and convergence of training.
    Soft Updates: The weights of the target networks are updated by blending them with the main networks' weights using a factor tau. This soft update technique contributes to the stability of the learning process.
"""
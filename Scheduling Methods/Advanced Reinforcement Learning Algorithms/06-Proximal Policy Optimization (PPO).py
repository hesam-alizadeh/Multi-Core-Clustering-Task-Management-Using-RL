import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gym
from collections import deque
import random
class PPOAgent:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 0.2
        self.lr = 0.0003
        self.batch_size = 64
        self.epochs = 10

        # Initialize actor and critic networks
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(self.lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.lr)

    def build_actor(self):
        inputs = layers.Input(shape=(self.state_dim,))
        out = layers.Dense(64, activation="relu")(inputs)
        out = layers.Dense(64, activation="relu")(out)
        outputs = layers.Dense(self.action_dim, activation="tanh")(out)
        model = tf.keras.Model(inputs, outputs)
        return model

    def build_critic(self):
        inputs = layers.Input(shape=(self.state_dim,))
        out = layers.Dense(64, activation="relu")(inputs)
        out = layers.Dense(64, activation="relu")(out)
        outputs = layers.Dense(1, activation=None)(out)
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        probabilities = self.actor(state)
        return np.clip(probabilities[0], -self.action_bound, self.action_bound)

    def compute_advantages(self, rewards, values):
        returns = []
        discounted_sum = 0
        for reward in rewards[::-1]:
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        advantages = np.array(returns) - values
        return advantages, returns

    def train(self, states, actions, advantages, returns):
        old_probabilities = self.actor(states)
        
        for _ in range(self.epochs):
            with tf.GradientTape() as tape:
                probabilities = self.actor(states)
                ratios = tf.exp(tf.math.log(probabilities) - tf.math.log(old_probabilities))
                surr1 = ratios * advantages
                surr2 = tf.clip_by_value(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            
            with tf.GradientTape() as tape:
                critic_loss = tf.reduce_mean(tf.square(returns - self.critic(states)))
            
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    def learn(self, env, max_episodes):
        for episode in range(max_episodes):
            state = env.reset()
            states, actions, rewards, values = [], [], [], []

            done = False
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                value = self.critic(np.reshape(state, [1, self.state_dim]))
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value[0])

                state = next_state

            values = np.array(values)
            advantages, returns = self.compute_advantages(rewards, values)
            self.train(np.array(states), np.array(actions), advantages, returns)

            print(f"Episode: {episode + 1}, Reward: {sum(rewards)}")


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    agent = PPOAgent(state_dim, action_dim, action_bound)
    agent.learn(env, max_episodes=1000)
"""
Explanation:
The Proximal Policy Optimization (PPO) algorithm is a powerful method within the family of policy gradient methods in reinforcement learning. Unlike standard policy gradient methods, PPO uses a clipping mechanism to constrain policy updates, thus preventing large deviations from the current policy. This constraint, controlled by a hyperparameter epsilon, ensures stable learning by balancing exploration and exploitation.
The PPOAgent class defines the key components of the PPO algorithm:
    Actor Network: The policy network responsible for selecting actions based on the current state.
    Critic Network: The value network estimates the expected return (or value) of being in a given state, used for advantage estimation.
    Advantage Calculation: The difference between the observed return and the estimated value is used to guide the policy updates, focusing on actions that yield higher-than-expected rewards.
    Policy Update: The actor network is updated to maximize the expected advantage while ensuring that the new policy does not deviate too much from the old policy, as measured by the ratio of probabilities.
"""
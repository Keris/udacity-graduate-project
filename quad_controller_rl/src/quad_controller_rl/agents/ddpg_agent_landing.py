import os
import random

from collections import namedtuple, deque

import numpy as np
import pandas as pd

from quad_controller_rl import util
from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.model import Actor, Critic


class OUNoise:
    '''Ornstein-Uhlenbeck process.'''

    def __init__(self, size, mu=None, theta=0.15, sigma=0.3):
        '''Initialize parameters and noise process.'''
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) + self.mu
        self.reset()

    def reset(self):
        '''Reset the internal state (= noise) to mean (mu).'''
        self.state = self.mu

    def sample(self):
        '''Update internal state and return it as a noise sample.'''
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size circular buffer to store experience tuples."""

    def __init__(self, size=1000):
        """Initialize a ReplayBuffer object."""
        self.size = size  # maximum size of buffer
        self.memory = []  # internal memory (list)
        self.idx = 0  # current index into circular buffer
        self.experience = namedtuple("Experience",
            field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # Create an Experience object, add it to memory
        # Note: If memory is full, start overwriting from the beginning
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.idx] = self.experience(state, action, reward, next_state, done)
        self.idx = (self.idx + 1) % self.size

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        # Return a list or tuple of Experience objects sampled from memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class DDPGAgentLanding(BaseAgent):
    '''Reinforcement Learning agent that learns using DDPG.'''
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = 7
        self.action_size = 3  # force only

        self.state_low = np.concatenate([
            self.task.observation_space.low[:3],
            np.array([0.0, 0.0, 0.0, 0.0])
        ])
        self.state_high = np.concatenate([
            self.task.observation_space.high[:3],
            self.task.observation_space.high[:3] - self.task.observation_space.low[:3],
            np.array([self.task.observation_space.high[2] - 10.0])
        ])
        self.state_range = self.state_high - self.state_low
        print("state low: {} state high: {} state range: {}".format(
            self.state_low, self.state_high, self.state_range))

        # clip action
        self.action_low = self.task.action_space.low[:self.action_size]
        self.action_high = self.task.action_space.high[:self.action_size]

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Intialize target model parameters with local model parameters
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())

        # Noise process
        self.noise = OUNoise(self.action_size)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters

        # Score tracker
        self.best_score = -np.inf

        # Episode variables
        self.reset_episode_vars()

        # Save episode stats
        self.stats_filename = os.path.join(
            util.get_param('out'),
            'landing/stats_{}.csv'.format(util.get_timestamp()))  # path to CSV file
        self.stats_columns = ['episode', 'total_reward']  # specify columns to save
        print('Saving stats {} to {}'.format(self.stats_columns, self.stats_filename))  # [debug]

        # Save weights
        self.save_weights_every = 100
        self.actor_filename = os.path.join(
            util.get_param('out'),
            'landing/actor_checkpoints_{}.h5'.format(util.get_timestamp())
        )
        self.critic_filename = os.path.join(
            util.get_param('out'),
            'landing/critic_checkpoints_{}.h5'.format(util.get_timestamp())
        )
        print('Actor filename: ', self.actor_filename)
        print('Critic filename: ', self.critic_filename)
        
        self.episode_num = 1
        self.reset_episode_vars()

    def write_stats(self, stats):
        '''Write single episode stats to CSV file.'''
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
            header=not os.path.isfile(self.stats_filename))  # write header first time only

    def preprocess_state(self, state):
        '''Reduce state vector to relevant dimensions.'''
        return state[:self.state_size]

    def postprocess_action(self, action):
        '''Return complete action vector.'''
        complete_action = np.zeros(self.task.action_space.shape)  # shape: (6, )
        complete_action[0:self.action_size] = action
        return complete_action

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0
        self.noise.reset()

    def step(self, state, reward, done):
        # Transform state vector
        # state = state.reshape(1, -1)  # convert to row vector
        state = self.preprocess_state(state)  # reduce state vector
        state = (state - self.state_low) / self.state_range  # scale to [0.0, 1.0]

        # Choose an action
        action = self.act(state)

        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done)
            self.total_reward += reward
            self.count += 1

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

        if done:
            # Write episode stats
            self.write_stats([self.episode_num, self.total_reward])
            if self.episode_num % self.save_weights_every == 0:
                print("Saving model weights... (episode_num: {})".format(self.episode_num))
                self.actor_local.model.save_weights(self.actor_filename)
                self.critic_local.model.save_weights(self.critic_filename)
            self.episode_num += 1

            score = self.total_reward / float(self.count) if self.count else 0.0
            if score > self.best_score:
                self.best_score = score
            print('DDPG: t = {:4d}, score = {:7.3f} (best = {:7.3f})'.format(self.count, score, self.best_score))  # [debug]
            self.reset_episode_vars()

        self.last_state = state
        self.last_action = action
        return self.postprocess_action(action)

    def act(self, states, add_noise=True):
        '''Return actions for given state(s) as per current policy.'''
        states = np.reshape(states, [-1, self.state_size])
        actions = self.actor_local.model.predict(states)
        if add_noise:
            actions += self.noise.sample()  # add some noise for exploration
        return actions

    def learn(self, experiences):
        '''Update policy and value parameters using given batch of experience tuples.'''
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        '''Soft update model parameters.'''
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

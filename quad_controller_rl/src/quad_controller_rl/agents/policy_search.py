"""Policy search agent."""
import os

import numpy as np
import pandas as pd
from quad_controller_rl import util
from quad_controller_rl.agents.base_agent import BaseAgent

class RandomPolicySearch(BaseAgent):
    """Sample agent that searches for optimal policy randomly."""

    def __init__(self, task):
        # Task (environment) information
        self.task = task  # should contain observation_space and action_space
        # self.state_size = np.prod(self.task.observation_space.shape)
        self.state_size = 3
        self.state_range = self.task.observation_space.high - self.task.observation_space.low

        self.action_size = 3
        # self.action_size = np.prod(self.task.action_space.shape)
        self.action_range = self.task.action_space.high - self.task.action_space.low

        # Policy parameters
        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range[:3] / (2 * self.state_size)).reshape(1, -1))  # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode_vars()

        # Save episode stats
        self.stats_filename = os.path.join(
            util.get_param('out'),
            '{}_{}_stats_{}.csv'.format(self.task, self, util.get_timestamp()))  # path to CSV file
        self.stats_columns = ['episode', 'total_reward']  # specify columns to save
        self.episode_num = 1
        print('Saving stats {} to {}'.format(self.stats_columns, self.stats_filename))  # [debug]

    def write_stats(self, stats):
        '''Write single episode stats to CSV file.'''
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
            header=not os.path.isfile(self.stats_filename))  # write header first time only

    def preprocess_state(self, state):
        '''Reduce state vector to relevant dimensions.'''
        return state[:, :3] # position only

    def postprocess_action(self, action):
        '''Return complete action vector.'''
        complete_action = np.zeros(self.task.action_space.shape)  # shape: (6, )
        complete_action[0:3] = action
        return complete_action

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0
        self.noise_scale = 0.1

    def step(self, state, reward, done):
        # Transform state vector
        state = (state - self.task.observation_space.low) / self.state_range  # scale to [0.0, 1.0]
        state = state.reshape(1, -1)  # convert to row vector
        state = self.preprocess_state(state)

        # Choose an action
        action = self.act(state)

        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.total_reward += reward
            self.count += 1

        # Learn, if at end of episode
        if done:
            # Write episode stats
            self.write_stats([self.episode_num, self.total_reward])
            self.episode_num += 1
            self.learn()
            self.reset_episode_vars()

        self.last_state = state
        self.last_action = action
        return self.postprocess_action(action)

    def act(self, state):
        # Choose action based on given state and policy
        action = np.dot(state, self.w)  # simple linear policy
        if self.count % 20 == 0:
            print(action)  # [debug: action vector]
        return action

    def learn(self):
        # Learn by random policy search, using a reward-based score
        score = self.total_reward / float(self.count) if self.count else 0.0
        if score > self.best_score:
            self.best_score = score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)

        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
        print("RandomPolicySearch.learn(): t = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                self.count, score, self.best_score, self.noise_scale))  # [debug]
        #print(self.w)  # [debug: policy parameters]

    def __repr__(self):
        return self.__class__.__name__

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from .pbenvs import make

import os

import numpy as np
import gym


class HalfCheetahEnv(gym.Env):
    OBSERVATION_SIZE = 27
    ACTION_SIZE = 6

    def __init__(self):
        self._env = make("HalfCheetahBulletEnv")
        self._prev_pos = None

        self.action_space = self._env.action_space
        self.observation_space = gym.spaces.Box(
            low = np.concatenate([np.array([-np.inf]), self._env.observation_space.low]),
            high = np.concatenate([np.array([np.inf]), self._env.observation_space.high]),
        )

    def step(self, action):
        self._prev_pos = self._env.robot_body.current_position()
        _next_obs, _, _, _ = self._env.step(action)

        pos = self._env.robot_body.current_position()
        next_obs = np.concatenate([pos[:1] - self._prev_pos[:1], _next_obs])

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = next_obs[0]
        reward = reward_run + reward_ctrl

        done = False
        return next_obs, reward, done, {}

    def reset(self):
        _obs = self._env.reset()
        pos = self._env.robot_body.current_position()
        self._prev_pos = pos
        return np.concatenate([pos[:1] - self._prev_pos[:1], _obs])

    def render(self, mode='human'):
        return self._env.render(mode)

    def seed(self, seed=None):
        pass


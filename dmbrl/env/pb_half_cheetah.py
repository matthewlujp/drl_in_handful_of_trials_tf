from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from .pbenvs import make

import os

import numpy as np
import gym


class HalfCheetahEnv(gym.Env):
    def __init__(self):
        self._env = make("HalfCheetahBulletEnv")
        self._prev_pos = None

        self.action_space = self._env.action_space
        self.observation_space = gym.spaces.Box(
            low = np.concatenate([np.array([-np.inf]), self._env.observation_space.low]),
            high = np.concatenate([np.array([np.inf]), self._env.observation_space.high]),
        )
        # self.observation_space = self._env.observation_space

    def step(self, action):
        self._prev_pos = self._env.robot.body_xyz
        _next_obs, r, _, _ = self._env.step(action)

        pos = self._env.robot.body_xyz
        next_obs = np.concatenate([np.array([pos[0] - self._prev_pos[0]]), _next_obs])

        # reward_ctrl = -00.1 * np.square(action).sum()
        # reward_run = next_obs[0]
        # reward = reward_run + reward_ctrl

        # return next_obs, reward, False, {}
        return next_obs, r, False, {}

    def reset(self):
        _obs = self._env.reset()
        pos = self._env.robot.body_xyz
        self._prev_pos = pos
        return np.concatenate([np.array([pos[0] - self._prev_pos[0]]), _obs])

    def render(self, mode='human'):
        return self._env.render(mode)

    def seed(self, seed=None):
        pass


import numpy as np
import torch

from .sensor import Sensor


class Interpreter():
    def __init__(self, frames=4, width=84, height=84):
        self.width = width
        self.height = height
        self.frames = frames

        self.state = np.zeros([self.frames, self.width, self.height])

    def obs_to_state(self, state, observation):
        sensor = Sensor(self.width, self.height)
        obs = torch.FloatTensor(sensor.preprocess_obs(observation))

        self.reset()

        if state is None:
            for i in range(self.frames):
                self.state[i, :, :] = obs
        else:
            self.state[:self.frames - 1, :, :] = state[1:, :, :]
            self.state[self.frames - 1, :, :] = obs

        return self.state

    def reset(self):
        self.state = np.zeros([self.frames, self.width, self.height])

    def process_reward(self, info, initial_health=176):
        health_gap = info['health'] - info['enemy_health']
        reward = float(health_gap / initial_health)

        return reward

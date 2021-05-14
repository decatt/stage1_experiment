import numpy as np


class Memory:
    def __init__(self, obs, action, reward, obs_):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.obs_ = obs_

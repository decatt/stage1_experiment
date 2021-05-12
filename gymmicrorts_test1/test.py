import time

import numpy as np
import torch
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
import microrts_agent

env = MicroRTSVecEnv(
    num_envs=1,
    render_theme=2,
    ai2s=[microrts_ai.passiveAI],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)
Epsilon = 0.3
obs = env.reset()
env.render()
env.action_space.seed(0)
s = env.reset()
ac = np.array([16*16, 6, 4, 4, 4, 4, 7, 16*16])


agent = microrts_agent.Agent(ac.sum())

episode_reward = 0
x = 1
y = 1
action = [0, 0, 0, 0, 0, 0, 0, 0]
for i in range(10000):
    env.render()
    action_mask = np.array(env.vec_client.getUnitLocationMasks()).flatten()

    time.sleep(0.02)

    if x <= 13:
        if s[0][y][x+1][17] == 1:
            x = x+1
        action = [16*y+x, 1, 1, 0, 0, 0, 0, 0]
    elif x == 14 and y < 13:
        if s[0][y+1][x][17] == 1:
            y = y+1
        action = [16*y+x, 1, 2, 0, 0, 0, 0, 0]
    else:
        action = [16 * y + x, 5, 0, 0, 0, 0, 0, 221]
    w = np.zeros((16, 16))
    # optional: selecting only valid units.
    s, reward, done, info = env.step([action])
    episode_reward = reward + episode_reward
    if done:
        print(str(episode_reward))
        episode_reward = 0
        env.reset()
env.close()
print('End')

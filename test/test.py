import time

import torch

import microrts_agent
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai


env = MicroRTSVecEnv(
    num_envs=1,
    render_theme=2,
    ai2s=[microrts_ai.passiveAI],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)
env.action_space.seed(0)
s = env.reset()
env.render()
ac = np.array([16*16, 6, 4, 4, 4, 4, 7, 16*16])


episode_reward = 0
action = [0, 0, 0, 0, 0, 0, 0, 0]
units = dict()
create_worker = True
x = 1
y = 49
m = 0
create = True
for i in range(10000):
    env.render()
    action_mask = np.array(env.vec_client.getUnitLocationMasks()).flatten()
    vu = []
    for j in range(256):
        if action_mask[j] == 1:
            vu.append(j)
    time.sleep(0.005)
    if s[0][1][1][17] == 1 and s[0][1][1][6] == 0 and s[0][1][0][5] == 0:
        action = [17, 2, 0, 3, 0, 0, 0, 0]
    elif s[0][1][1][6] == 1:
        action = [17, 1, 1, 0, 0, 0, 0, 0]
    elif s[0][1][2][6] == 1:
        action = [16 * 1 + 2, 3, 0, 0, 2, 0, 0, 0]
    elif s[0][1][2][6] == 0 and s[0][1][2][17] == 1:
        action = [16 * 1 + 2, 1, 3, 0, 0, 0, 0, 0]
    elif s[0][1][0][5] == 1 and s[0][2][1][16] == 0 and s[0][1][1][17] == 1:
        action = [17, 4, 0, 0, 0, 2, 2, 0]
    elif s[0][2][1][16] == 1 and vu.count(17) == 1:
        action = [17, 1, 0, 0, 0, 0, 0, 0]
    elif s[0][2][1][16] == 1 and vu.count(33) == 1 and create:
        action = [33, 4, 0, 0, 0, 2, 5, 0]
        create = False
    elif vu.count(x) == 1 and vu.count(49) < 1:
        if (m//10) % 2 == 0:
            action = [x, 1, 1, 0, 0, 0, 0, 0]
            x = x+1
        else:
            action = [x, 1, 3, 0, 0, 0, 0, 0]
            x = x - 1
        m = m+1
    elif vu.count(y) == 1:
        if y < 62:
            action = [y, 1, 1, 0, 0, 0, 0, 0]
            y = y+1
        elif y <= 206:
            action = [y, 1, 2, 0, 0, 0, 0, 0]
            y = y+16
    elif s[0][222//16][222 % 16][19] == 1:
            action = [y,5,0,0,0,0,0,221]
    else:
        action = [34, 0, 0, 0, 0, 0, 0, 0]
    s, reward, done, info = env.step([action])
    if done:
        env.reset()
env.close()
print('End')


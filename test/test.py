import time
from typing import Dict

import torch

import microrts_agent
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from jpype.types import JArray, JInt
from environment import MicroRTSStatsRecorder, VecMonitor
import map_parameter

env = MicroRTSVecEnv(
    num_envs=1,
    render_theme=2,
    ai2s=[microrts_ai.passiveAI],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)
mp = map_parameter.MapParameter(map_h=16, map_w=16)
env = MicroRTSStatsRecorder(env, 0.99)
env = VecMonitor(env)
env.action_space.seed(0)
s = env.reset()
env.render()

ac = np.array([16*16, 6, 4, 4, 4, 4, 7, 16*16])
m=0
episode_reward = 0
action = [0, 0, 0, 0, 0, 0, 0, 0]

create_worker = True
pos_worker1 = 1
pos_heavy1 = 49
pos_base = 34
action_types = {'noop': 0, 'move': 1, 'harvest': 2, 'return': 3, 'produce': 4, 'attack': 5}
a=0
create = True
for i in range(10000):
    env.render()
    action_mask = np.array(env.vec_client.getUnitLocationMasks()).flatten()
    vu = []
    for j in range(256):
        if action_mask[j] == 1:
            vu.append(j)
    time.sleep(0.03)
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

    elif vu.count(pos_worker1) == 1 and vu.count(49) < 1:
        if (m // 10) % 2 == 0:
            action = [pos_worker1, 1, 1, 0, 0, 0, 0, 0]
            pos_worker1 = pos_worker1 + 1
        else:
            action = [pos_worker1, 1, 3, 0, 0, 0, 0, 0]
            pos_worker1 = pos_worker1 - 1
        m = m + 1

    elif vu.count(pos_heavy1) == 1:
        if pos_heavy1 < 62:
            action = [pos_heavy1, 1, 1, 0, 0, 0, 0, 0]
            pos_heavy1 = pos_heavy1 + 1
        elif pos_heavy1 <= 206:
            action = [pos_heavy1, 1, 2, 0, 0, 0, 0, 0]
            pos_heavy1 = pos_heavy1 + 16
        elif s[0][222 // 16][222 % 16][19] == 1:
            x = 3 + (221 % 16) - (222 % 16)
            y = 3 + (221 // 16) - (222 // 16)
            action = [pos_heavy1, 5, 0, 0, 0, 0, 0, x + y*7]
    else:
        action = [pos_base, 0, 0, 0, 0, 0, 0, 0]
    valid_action = action
    java_valid_action = [JArray(JInt)(valid_action)]
    # java_valid_actions = JArray(JArray(JArray(JInt)))(java_valid_action)
    java_valid_actions = JArray(JArray(JInt))(java_valid_action)
    java_valid_actions = JArray(JArray(JInt))(java_valid_actions)
    try:
        s, reward, done, info = env.step(java_valid_actions)
    except Exception as e:
        e.printStackTrace()
        raise
    if done:
        env.reset()
env.close()
print('End')

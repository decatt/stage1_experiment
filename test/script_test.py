import time

import torch
import microrts_script
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
action = [34, 0, 0, 0, 0, 0, 0, 0]
units = dict()
create_worker = True
x = 1
y = 49
m = 0
create = True
worker = microrts_script.ScriptUnit((1, 1), 17)
worker.target = 16
base = microrts_script.ScriptUnit((2, 2), 15)
base.target = 16*3+2
base.state = 1
for i in range(10000):
    action = [34,0,0,0,0,0,0,0]
    env.render()
    action_mask = np.array(env.vec_client.getUnitLocationMasks()).flatten()
    vu = []
    for j in range(256):
        if action_mask[j] == 1:
            vu.append(j)
    time.sleep(0.005)

    if vu.count(16*worker.y+worker.x) == 1:
        action = worker.harvest(s)
    if vu.count(16*base.y+base.x) == 1 and base.state == 1:
        action = base.produce(s)

    s, reward, done, info = env.step([action])
    if vu.count(16*3+2) == 1:
        print('created')
    if done:
        env.reset()
env.close()
print('End')

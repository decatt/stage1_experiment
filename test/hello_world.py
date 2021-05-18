import numpy as np
import time
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai

env = MicroRTSVecEnv(
    num_envs=1,
    render_theme=2,
    ai2s=[microrts_ai.coacAI],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)
obs = env.reset()
env.render()
env.action_space.seed(0)
env.reset()

for i in range(10000):
    env.render()
    action_mask = np.array(env.vec_client.getUnitLocationMasks()).flatten()
    time.sleep(0.001)
    action = env.action_space.sample()
    # optional: selecting only valid units.
    if len(action_mask.nonzero()[0]) != 0:
        action[0] = action_mask.nonzero()[0][0]

    next_obs, reward, done, info = env.step([action])
    if done:
        env.reset()
env.close()

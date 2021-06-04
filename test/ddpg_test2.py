import torch
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
import ddpg
import microrts_agent
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


env = MicroRTSVecEnv(
    num_envs=1,
    render_theme=2,
    ai2s=[microrts_ai.workerRushAI],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    frame_skip=20
)


env.action_space.seed(0)
s = env.reset()
env.render()

# env = VecMonitor(env)
# env = VecPyTorch(env, device)
action_space = np.array([16*16, 6, 4, 4, 4, 4, 7, 7*7])
N_actions = action_space.sum()
N_states = 16*16*27
Epsilon = 0.9

ddpg_model = ddpg.DDPG(N_states, N_actions)

rewards = []
action_f = [0, 0, 0, 0, 0, 0]

for i in range(1000):
    s = env.reset()
    ep_reward = 0
    step = 0
    while True:
        action_mask = np.array(env.vec_client.getUnitLocationMasks()).flatten()
        env.render()
        action = [34, 0, 0, 0, 0, 0, 0, 0]
        vu = []
        for j in range(256):
            if action_mask[j] == 1:
                vu.append(j)

        if len(vu) >= 1:
            unit = random.sample(vu, 1)
            if np.random.uniform() < Epsilon:
                action = []
                try:
                    acts = ddpg_model.Actor_eval.get_action(torch.Tensor(s), action_space, env)
                except:
                    i = i-1
                    break
                else:
                    for a in acts:
                        action.append(a[0])

            else:
                try:
                    actions = microrts_agent.get_valid_actions(action_space, env, unit)
                except:
                    i = i-1
                    break
                else:
                    if len(actions) > 0:
                        # select a low frequency action
                        act_dict = [1e10, 1e10, 1e10, 1e10, 1e10, 1e10]
                        for a in actions:
                            act_dict[a[1]] = action_f[a[1]]
                        action_type = act_dict.index(min(act_dict))
                        action = random.sample([x for x in actions if x[1] == action_type], 1)[0]
        else:
            action = [34,0,0,0,0,0,0,0]
        action_type = action[1]
        action_f[action[1]] = action_f[action[1]] + 1
        step = step + 1
        s_, r, done, info = env.step([action])

        ddpg_model.store_transition(s, microrts_agent.action_encoder(action), r / 10, s_)

        if ddpg_model.pointer >= 2000 and (ddpg_model.pointer % 500) == 0:
            ddpg_model.learn()
        if (ddpg_model.pointer % 4096) == 0:
            action_f = [0, 0, 0, 0, 0, 0]

        s = s_
        ep_reward += r
        if done:
            print('Ep: ', i, '| Ep_r: ', ep_reward)
            rewards.append(ep_reward)
            torch.save(ddpg_model.Actor_eval, './microrts_ddpg_actor.pth')
            torch.save(ddpg_model.Critic_eval, './microrts_ddpg_critic.pth')
            break

print('End')
print(rewards)
print(rewards[:-1])

fig = plt.figure()

data = rewards
y = np.array(data)
x = np.array(range(len(data)))

plt.scatter(x, y, color='red', marker='+')
plt.show()

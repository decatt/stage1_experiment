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
    ai2s=[microrts_ai.coacAI],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    frame_skip=20
)

env.action_space.seed(0)
env.render()

# env = VecMonitor(env)
# env = VecPyTorch(env, device)
action_space = np.array([16*16, 6, 4, 4, 4, 4, 7, 16*16])
N_actions = action_space.sum()
N_states = 16*16*27
Epsilon = 0.9

ddpg_model = ddpg.DDPG(N_states, N_actions)

rewards = []
action_f = [0, 0, 0, 0, 0, 0]

print('start')
start_time = time.time()
while True:
    s = env.reset()
    ep_reward = 0
    step = 0
    while True:
        env.render()
        action = [34, 0, 0, 0, 0, 0, 0, 0]
        action_mask = np.array(env.vec_client.getUnitLocationMasks()).flatten()
        vu = []
        for j in range(256):
            if action_mask[j] == 1:
                vu.append(j)
        if len(vu) >= 1:
            if np.random.uniform() < Epsilon:
                action = []
                acts = ddpg_model.Actor_eval.get_action(torch.Tensor(s).to(device), action_space, env)
                for a in acts:
                    action.append(a[0])

            else:
                unit = random.sample(vu, 1)
                actions = microrts_agent.get_valid_actions(action_space, env, unit)
                if len(actions) > 0:
                    action = random.sample(actions, 1)[0]

        # transform action
        new_action = action.copy()
        pux = action[0] % 16
        puy = action[0] // 16
        pax = action[7] % 16
        pay = action[7] // 16
        new_attack_pos = 3 + pax - pux + 7*(3 + pay - puy)
        new_action[7] = new_attack_pos

        step = step + 1
        s_, r, done, info = env.step([new_action])

        ddpg_model.store_transition(s, microrts_agent.action_encoder(action), r, s_)

        if ddpg_model.pointer >= 10000 and (ddpg_model.pointer % 100) == 0:
            ddpg_model.learn()

        s = s_
        ep_reward += r
        if done or step >= 10000:
            print('Ep_r: ', ep_reward)
            rewards.append(ep_reward)
            torch.save(ddpg_model.Actor_eval, './microrts_ddpg_actor.pth')
            torch.save(ddpg_model.Critic_eval, './microrts_ddpg_critic.pth')
            break

    if (time.time() - start_time) > 8 * 60 * 60:
        break

print('End')
print(rewards)
print(rewards[:-1])

fig = plt.figure()

data = rewards
y = np.array(data)
x = np.array(range(len(data)))

plt.plot(x, y, color='red')
plt.show()

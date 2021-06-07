import torch
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
import ddpg
import microrts_agent
import random
import matplotlib.pyplot as plt
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def m_loop(envi, obs):
    action = [34, 0, 0, 0, 0, 0, 0, 0]
    action_mask = np.array(envi.vec_client.getUnitLocationMasks()).flatten()
    vu = []
    for j in range(256):
        if action_mask[j] == 1:
            vu.append(j)
    if len(vu) >= 1:
        if np.random.uniform() < Epsilon:
            action = []
            acts = ddpg_model.Actor_eval.get_action(torch.Tensor(obs).to(device), action_space, envi)
            for a in acts:
                action.append(a[0])

        else:
            unit = random.sample(vu, 1)
            actions = microrts_agent.get_valid_actions(action_space, envi, unit)
            if len(actions) > 0:
                action = random.sample(actions, 1)[0]

    # transform action
    new_action = action.copy()
    pux = action[0] % 16
    puy = action[0] // 16
    pax = action[7] % 16
    pay = action[7] // 16
    new_attack_pos = 3 + pax - pux + 7 * (3 + pay - puy)
    new_action[7] = new_attack_pos

    s_1, r1, done1, info1 = envi.step([new_action])

    ddpg_model.store_transition(obs, microrts_agent.action_encoder(action), r1, s_1)

    return s_1, r1, done1, info1


env = MicroRTSVecEnv(
    num_envs=1,
    render_theme=2,
    ai2s=[microrts_ai.coacAI],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    frame_skip=20
)

env2 = MicroRTSVecEnv(
    num_envs=1,
    render_theme=2,
    ai2s=[microrts_ai.coacAI],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    frame_skip=20
)

env3 = MicroRTSVecEnv(
    num_envs=1,
    render_theme=2,
    ai2s=[microrts_ai.coacAI],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    frame_skip=20
)

env.action_space.seed(0)
s = env.reset()

env2.action_space.seed(0)
s2 = env2.reset()

env3.action_space.seed(0)
s3 = env3.reset()

action_space = np.array([16 * 16, 6, 4, 4, 4, 4, 7, 16 * 16])
N_actions = action_space.sum()
N_states = 16 * 16 * 27
Epsilon = 0.9

ddpg_model = ddpg.DDPG(N_states, N_actions)

rewards = []
action_f = [0, 0, 0, 0, 0, 0]

print('start')
start_time = time.time()
ep_reward = 0
step = 0
ep_reward2 = 0
step2 = 0
ep_reward3 = 0
step3 = 0
while True:
    env.render()
    # env2.render()
    # env3.render()

    s_, r, done, info = m_loop(env, s)
    step = step + 1
    s = s_
    ep_reward += r

    s_2, r2, done2, info2 = m_loop(env2, s2)
    step2 = step2 + 1
    s2 = s_2
    ep_reward2 += r2

    s_3, r3, done3, info3 = m_loop(env3, s3)
    step3 = step3 + 1
    s3 = s_3
    ep_reward3 += r3

    if ddpg_model.pointer >= 30000 and (ddpg_model.pointer % 100) == 0:
        ddpg_model.learn()

    if done or step >= 10000:
        print('Ep_r: ', ep_reward)
        rewards.append(ep_reward)
        torch.save(ddpg_model.Actor_eval, './microrts_ddpg_actor_2env_3h.pth')
        torch.save(ddpg_model.Critic_eval, './microrts_ddpg_critic_2env_3h.pth')
        ep_reward = 0
        step = 0
        s = env.reset()

    if done2 or step >= 10000:
        s2 = env2.reset()
        rewards.append(ep_reward2)
        ep_reward2 = 0
        step2 = 0

    if done3 or step >= 10000:
        s3 = env3.reset()
        rewards.append(ep_reward2)
        ep_reward3 = 0
        step3 = 0

    if (time.time() - start_time) > 1 * 60 * 60:
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

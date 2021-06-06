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

env_number = 1

env = MicroRTSVecEnv(
    num_envs=env_number,
    render_theme=2,
    ai2s=[microrts_ai.coacAI for _ in range(env_number)],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([100.0, 1.0, 1.0, 0.2, 10.0, 4.0]),
    frame_skip=20
)

env.action_space.seed(0)
s = env.reset()
env.render()

# env = VecMonitor(env)
# env = VecPyTorch(env, device)
action_space = np.array([16*16, 6, 4, 4, 4, 4, 7, 16*16])
N_actions = action_space.sum()
N_states = 16*16*27
Epsilon = 1

ddpg_model = ddpg.DDPG(N_states, N_actions)

rewards = []
action_f = [0, 0, 0, 0, 0, 0]
t_step = 0

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
        factions = []
        vus = []
        for j in range(env_number):
            vu = []
            for k in range(256):
                if action_mask[k*(j+1)] == 1:
                    vu.append(k)
            vus.append(vu)
        if len(vus) >= 1 and len(vus[0]) >= 1:
            if np.random.uniform() >= Epsilon:
                action = []
                acts = ddpg_model.Actor_eval.get_action(torch.Tensor(s).to(device), action_space, env)
                for a in acts:
                    action.append(a[0])

            else:
                units = []
                for n in range(len(vus)):
                    unit = random.sample(vus[n], 1)
                    actions = microrts_agent.get_valid_actions(action_space, env, unit, n)
                    if len(actions) > 0:
                        action = random.sample(actions, 1)[0]
                    factions.append(action)

        # transform action
        new_actions = []
        for action in factions:
            new_action = action.copy()
            pux = action[0] % 16
            puy = action[0] // 16
            pax = action[7] % 16
            pay = action[7] // 16
            new_attack_pos = 3 + pax - pux + 7*(3 + pay - puy)
            new_action[7] = new_attack_pos
            new_actions.append(new_action)
        step = step + 1
        s_, r, done, info = env.step(new_actions)

        if s_[0][221 // 16][221 % 16][15] == 0:
            r = 100

        ddpg_model.store_transition(s, microrts_agent.action_encoder(action), r, s_)

        if ddpg_model.pointer >= 10000 and (ddpg_model.pointer % 100) == 0:
            ddpg_model.learn()

        s = s_
        ep_reward += r[0]
        if done or step >= 10000:
            t_step = t_step+step
            print('total step:', t_step, ' |Ep_r: ', ep_reward)
            rewards.append(ep_reward)
            torch.save(ddpg_model.Actor_eval, './microrts_ddpg_actor_0605_8h.pth')
            torch.save(ddpg_model.Critic_eval, './microrts_ddpg_critic_0605_8h.pth')
            Epsilon = Epsilon*0.99
            break

    if (time.time() - start_time) > 8*60*60:
        break

print('End')
print('rewards:', rewards)
fig = plt.figure()

data = rewards
y = np.array(data)
x = np.array(range(len(data)))

plt.plot(x, y, color='red')
plt.show()

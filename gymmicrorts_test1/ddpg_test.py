import torch
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
import ddpg
import microrts_agent
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

env = MicroRTSVecEnv(
    num_envs=1,
    render_theme=2,
    ai2s=[microrts_ai.coacAI],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)

env.action_space.seed(0)
s = env.reset()
env.render()
action_space = np.array([16*16, 6, 4, 4, 4, 4, 7, 16*16])
N_actions = action_space.sum()
N_states = 16*16*27
Epsilon = 0.9

ddpg_model = ddpg.DDPG(N_states, N_actions)

rewards = []

for i in range(1000):
    s = env.reset()
    ep_reward = 0
    step = 0
    while True:

        env.render()
        action = [0, 0, 0, 0, 0, 0, 0, 0]
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
                actions = microrts_agent.get_valid_actions(action_space, env)
            except:
                i = i-1
                break
            else:
                if len(actions) > 0:
                    action = random.sample(actions, 1)[0]
        step = step + 1
        s_, r, done, info = env.step([action])

        if step >= 1500 and not done:
            r = -100

        ddpg_model.store_transition(s, microrts_agent.action_encoder(action), r / 10, s_)

        if ddpg_model.pointer > 2000:
            ddpg_model.learn()

        s = s_
        ep_reward += r
        if done or step >= 1500:
            print('Ep: ', i, '| Ep_r: ', ep_reward)
            rewards.append(ep_reward)
            torch.save(ddpg_model.Actor_eval, './microrts_ddpg_actor.pth')
            torch.save(ddpg_model.Critic_eval, './microrts_ddpg_critic.pth')
            break

print('End')
print(rewards)

fig = plt.figure()

data = rewards
y = np.array(data)
x = np.array(range(len(data)))

plt.scatter(x, y, color='red', marker='+')
plt.show()

import torch
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
import dqn
import microrts_agent
import random

# parameters
Batch_size = 32
Lr = 0.01
Epsilon = 0.9  # greedy policy
Gamma = 0.9  # reward discount
Target_replace_iter = 100  # target update frequency
Memory_capacity = 2000
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

ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = './microrts_dqn_model.pth'

if __name__ == '__main__':
    print(torch.cuda.is_available())
    dqn_model = dqn.DQN(N_states, action_space)

    print('\nCollecting experience...')
    for i_episode in range(1000):
        s = env.reset()
        ep_r = 0
        while True:
            env.render()
            action = [0, 0, 0, 0, 0, 0, 0, 0]
            if np.random.uniform() < Epsilon:
                action = []
                acts = dqn_model.eval_net.get_action(torch.Tensor(s), action_space, env).to('cpu')
                for a in acts:
                    action.append(a[0])
            else:
                actions = microrts_agent.get_valid_actions(action_space, env)
                if len(actions) > 0:
                    action = random.sample(actions, 1)[0]

            # take action
            s_, r, done, info = env.step([action])

            dqn_model.store_transition(s, action, r, s_)
            ep_r = ep_r+r
            if dqn_model.memory_counter > Memory_capacity:
                dqn_model.learn()
                if done:
                    print('Ep: ', i_episode, '| Ep_r: ', ep_r)

            if done:
                torch.save(dqn_model.eval_net, PATH)
                break
            s = s_
    torch.save(dqn_model.eval_net, PATH)
    print('finish')
    env.close()

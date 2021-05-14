import random
import time
import torch
import torch.nn as nn
import numpy as np
import microrts_agent
import collections
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG(object):
    def __init__(self, n_states, action_space):
        self.n_action, self.n_states = action_space.sum(), n_states
        self.memory_capacity = 2000
        self.memory = collections.deque(maxlen=2000)
        self.pointer = 0
        self.Actor_eval = microrts_agent.Agent(action_space.sum())
        self.Actor_target = microrts_agent.Agent(action_space.sum())
        self.Critic_eval = microrts_agent.CriticNet()
        self.Critic_target = microrts_agent.CriticNet()
        self.c_train = torch.optim.Adam(self.Critic_eval.parameters(), lr=0.001)
        self.a_train = torch.optim.Adam(self.Actor_eval.parameters(), lr=0.002)
        self.loss_td = nn.MSELoss()
        self.learn_step_counter = 0  # for target updating
        self.target_replace_iter = 100  # target update frequency

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.Actor_eval(s)[0].detach()  # ae（s）

    def learn(self):
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.Actor_eval.load_state_dict(self.Actor_target.state_dict())
            self.Critic_eval.load_state_dict(self.Critic_target.state_dict())
        self.learn_step_counter += 1

        # soft target replacement
        # self.sess.run(self.soft_replace)  # 用ae、ce更新at，ct

        data = random.sample(self.memory, 128)
        bs = torch.FloatTensor(np.array([d[0] for d in data]).reshape((128, 16, 16, 27)))
        ba = torch.FloatTensor(np.array([d[1] for d in data]))
        br = torch.FloatTensor(np.array([d[2] for d in data]))
        bs_ = torch.FloatTensor(np.array([d[3] for d in data]).reshape((128, 16, 16, 27)))

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs, a)  # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q)
        # print(q)
        # print(loss_a)
        self.a_train.zero_grad()
        loss_a.backward()
        self.a_train.step()

        a_ = self.Actor_target(bs_)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(bs_, a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = br + 0.9 * q_  # q_target = 负的
        # print(q_target)
        q_v = self.Critic_eval(bs, ba)
        # print(q_v)
        td_error = self.loss_td(q_target, q_v)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        # print(td_error)
        self.c_train.zero_grad()
        td_error.backward()
        self.c_train.step()

    def store_transition(self, s, a, r, s_):
        self.memory.append((s, a, r, s_))
        self.pointer += 1

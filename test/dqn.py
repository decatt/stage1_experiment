import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import microrts_agent as ma


class DQN(object):
    def __init__(self, n_states, action_space):
        self.batch_size = 32
        self.lr = 0.01
        self.epsilon = 0.9  # greedy policy
        self.gamma = 0.9  # reward discount
        self.target_replace_iter = 100  # target update frequency
        self.memory_capacity = 2000
        self.eval_net, self.target_net = ma.Agent(action_space.sum()), ma.Agent(action_space.sum())
        self.action_space = action_space
        self.n_actions = action_space.sum()
        self.n_states = n_states
        # self.eval_net = torch.load(PATH).to(device)
        # self.target_net = torch.load(PATH).to(device)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((self.memory_capacity, n_states * 2 + 542))  # initialize memory
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_):
        sr = s.reshape(-1)
        sr_ = s_.reshape(-1)
        transition = np.hstack((sr, a, r, sr_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net update
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :self.n_states].reshape(self.batch_size, 16, 16, 27)))
        b_a = Variable(torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.n_states:].reshape(self.batch_size, 16, 16, 27)))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

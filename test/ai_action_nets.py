import gym.spaces
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

MapX = 16
MapY = 16


class PosNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(MapX * MapY * 27, MapX * MapY * 9)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(MapX * MapY * 9, MapX * MapY * 3)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(MapX * MapY * 3, MapX * MapY)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class ActTypeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(MapX * MapY * 27, 512)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(512, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(32, 6)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DirectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(MapX * MapY * 27, 512)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(512, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(32, 4)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class ProduceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(MapX * MapY * 27, 512)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(512, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(32, 7)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class AttackNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(MapX * MapY * 27, MapX * MapY * 9)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(MapX * MapY * 9, MapX * MapY * 3)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(MapX * MapY * 3, MapX * MapY)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


def get_action(obs, pn, an, dn, prn, atn):
    fobs = Variable(torch.unsqueeze(torch.FloatTensor(np.array(obs).flatten()), 0))
    pos = np.argmax(pn.forward(fobs).detach().numpy())
    action_type = np.argmax(an.forward(fobs).detach().numpy())
    direction = np.argmax(dn.forward(fobs).detach().numpy())
    action = [pos, action_type, 0, 0, 0, 0, 0, 0]
    if action_type == 0:
        action = [pos, action_type, 0, 0, 0, 0, 0, 0]
    elif 1 <= action_type <= 3:
        action[action_type+1] = direction
    elif action_type == 4:
        action[5] = direction
        action[6] = np.argmax(prn.forward(fobs).detach().numpy())
    elif action == 5:
        action[7] = get_attack_target(pos, direction)
    res = np.array(action)
    return res


def get_attack_target(pos, direction):
    res = 0
    if direction == 0:
        res = pos - MapX
    elif direction == 1:
        res = pos + 1
    elif direction == 2:
        res = pos + MapX
    elif direction == 3:
        res = pos - 1
    if res < 0 or res > (MapX * MapY - 1):
        return 0
    return res


def get_random_action():
    return gym.spaces.MultiDiscrete([MapX * MapY, 6, 4, 4, 4, 4, 7, MapX * MapY]).sample()

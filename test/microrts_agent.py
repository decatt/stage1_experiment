import gym.spaces
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from torch.distributions.categorical import Categorical


# ALGO LOGIC: initialize agent here:
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.))
        return -p_log_p.sum(-1)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, action_space):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 6 * 6, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_space), std=0.01))

    def forward(self, x):
        return self.network(x.permute((0, 3, 1, 2)))  # "bhwc" -> "bchw"

    def get_action(self, x, action_space, envs):
        logits = self.forward(x).to('cpu')
        split_logits = torch.split(logits, action_space.tolist(), dim=1)
        # 1. select source unit based on source unit mask
        source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(1, -1))
        multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
        action_components = [multi_categoricals[0].sample()]
        # 2. select action type and parameter section based on the
        #    source-unit mask of action type and parameters
        # print(np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(args.num_envs, -1))
        source_unit_action_mask = torch.Tensor(
            np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
        split_suam = torch.split(source_unit_action_mask, action_space.tolist()[1:], dim=1)
        multi_categoricals = multi_categoricals + [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                                   zip(split_logits[1:], split_suam)]
        action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
        action = torch.stack(action_components)
        return action


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 6 * 6, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1))

    def forward(self, x):
        return self.network(x.permute((0, 3, 1, 2)))


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.network1 = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 6 * 6, 256))
        )
        self.network2 = nn.Sequential(
            layer_init(nn.Linear(541, 256))
        )
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x, a):
        x = self.network1(x.permute((0, 3, 1, 2)))
        y = self.network2(a)
        c = F.relu(x+y)
        res = self.critic(c)
        return res


def get_valid_actions(action_space, envs, unit):
    actions = []
    i = unit[0]
    source_unit_action_mask = torch.Tensor(np.array(envs.vec_client.getUnitActionMasks(unit)))
    ats, mps, hps, rps, pdps, ptps, atus = torch.split(source_unit_action_mask, action_space.tolist()[1:], dim=1)
    ats = ats.tolist()[0]
    mps = mps.tolist()[0]
    hps = hps.tolist()[0]
    rps = rps.tolist()[0]
    pdps = pdps.tolist()[0]
    ptps = ptps.tolist()[0]
    atus = atus.tolist()[0]
    for at_i in range(len(ats)):
        if ats[at_i] == 1:
            if at_i == 0:
                action = [i, at_i, 0, 0, 0, 0, 0, 0]
                actions.append(action)
            elif at_i == 1:  # move
                for mp_i in range(len(mps)):
                    if mps[mp_i] == 1:
                        action = [i, at_i, mp_i, 0, 0, 0, 0, 0]
                        actions.append(action)
            elif at_i == 2:  # harvest
                for hp_i in range(len(hps)):
                    if mps[hp_i] == 1:
                        action = [i, at_i, 0, hp_i, 0, 0, 0, 0]
                        actions.append(action)
            elif at_i == 3:  # return
                for rp_i in range(len(rps)):
                    if rps[rp_i] == 1:
                        action = [i, at_i, 0, 0, rp_i, 0, 0, 0]
                        actions.append(action)
            elif at_i == 4:  # produce
                for pdp_i in range(len(pdps)):
                    if pdps[pdp_i] == 1:
                        for ptp_i in range(len(ptps)):
                            if ptps[ptp_i] == 1:
                                action = [i, at_i, 0, 0, 0, pdp_i, ptp_i, 0]
                                actions.append(action)
            elif at_i == 5:  # attack
                for atu_i in range(len(atus)):
                    if atus[atu_i] == 1:
                        action = [i, at_i, 0, 0, 0, 0, 0, atu_i]
                        actions.append(action)
    return actions


def action_encoder(action):
    pos = [0.]*256
    ats = [0.]*6
    mps = [0.]*4
    hps = [0.]*4
    rps = [0.]*4
    pdps = [0.]*4
    ptps = [0.]*7
    atus = [0.]*256
    pos[action[0]] = 1.
    ats[action[1]] = 1.
    mps[action[2]] = 1.
    hps[action[3]] = 1.
    rps[action[4]] = 1.
    pdps[action[5]] = 1.
    ptps[action[6]] = 1.
    atus[action[7]] = 1.
    return pos+ats+mps+hps+rps+pdps+ptps+atus


def transform_attack(action):
    unit_pos = action[0]
    attack_pos = action[7]
    new_attack_pos = attack_pos - unit_pos + 24
    action[7] = new_attack_pos
    return action

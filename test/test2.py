import time
import microrts_agent
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder
import torch


class VecMonitor(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                info['episode'] = epinfo
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class MicroRTSStatsRecorder(VecEnvWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        for i in range(len(dones)):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]
        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                raw_rewards = np.array(self.raw_rewards[i]).sum(0)
                raw_names = [str(rf) for rf in self.rfs]
                info['microrts_stats'] = dict(zip(raw_names, raw_rewards))
                self.raw_rewards[i] = []
                newinfos[i] = info
        return obs, rews, dones, newinfos


envs = MicroRTSVecEnv(
    num_envs=1,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.passiveAI for _ in range(1)],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
envs = MicroRTSStatsRecorder(envs, 0.99)
envs = VecMonitor(envs)
envs = VecPyTorch(envs, device)

envs.action_space.seed(0)
s = envs.reset()
envs.render()
ac = np.array([16*16, 6, 4, 4, 4, 4, 7, 16*16])


episode_reward = 0
x = 1
y = 1
action = [34, 0, 0, 0, 0, 0, 0, 0]
units = dict()
create_worker = True
for i in range(10000):
    envs.render()
    action_mask = np.array(envs.vec_client.getUnitLocationMasks()).flatten()
    vu = []
    for j in range(256):
        if action_mask[j] == 1:
            vu.append(j)
    time.sleep(0.005)


    s, reward, done, info = envs.step(torch.LongTensor([action]).to(device))
    episode_reward = reward + episode_reward
    if done:
        episode_reward = 0
        envs.reset()
envs.close()
print('End')

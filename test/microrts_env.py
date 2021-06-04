import os
import json
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from gym_microrts.envs.vec_env import MicroRTSVecEnv
import gym
import gym_microrts
from gym_microrts import microrts_ai

import jpype
from jpype.imports import registerDomain
import jpype.imports
from jpype.types import JArray


class MicrortsEnv(MicroRTSVecEnv):
    def __init__(self, utt, num_envs=2, max_steps=2000, render_theme=2, frame_skip=0,
                 ai2s=[microrts_ai.passiveAI, microrts_ai.passiveAI], map_path="maps/10x10/basesTwoWorkers10x10.xml",
                 reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 5.0])):

        super().__init__(num_envs, max_steps, render_theme, frame_skip, ai2s, map_path, reward_weight)
        assert num_envs == len(ai2s), "for each environment, a microrts ai should be provided"
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.render_theme = render_theme
        self.frame_skip = frame_skip
        self.ai2s = ai2s
        self.map_path = map_path
        self.reward_weight = reward_weight

        # read map
        self.microrts_path = os.path.join(gym_microrts.__path__[0], 'microrts')
        root = ET.parse(os.path.join(self.microrts_path, self.map_path)).getroot()
        self.height, self.width = int(root.get("height")), int(root.get("width"))

        # launch the JVM
        if not jpype._jpype.isStarted():
            registerDomain("ts", alias="tests")
            registerDomain("ai")
            jars = [
                "microrts.jar", "Coac.jar", "Droplet.jar", "GRojoA3N.jar",
                "Izanagi.jar", "MixedBot.jar", "RojoBot.jar", "TiamatBot.jar", "UMSBot.jar"  # "MindSeal.jar"
            ]
            for jar in jars:
                jpype.addClassPath(os.path.join(self.microrts_path, jar))
            jpype.startJVM(convertStrings=False)

        # start microrts client
        from rts.units import UnitTypeTable
        self.real_utt = UnitTypeTable()
        from ai.rewardfunction import RewardFunctionInterface, WinLossRewardFunction, ResourceGatherRewardFunction, \
            AttackRewardFunction, ProduceWorkerRewardFunction, ProduceBuildingRewardFunction, \
            ProduceCombatUnitRewardFunction, CloserToEnemyBaseRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([
            WinLossRewardFunction(),
            ResourceGatherRewardFunction(),
            ProduceWorkerRewardFunction(),
            ProduceBuildingRewardFunction(),
            AttackRewardFunction(),
            ProduceCombatUnitRewardFunction(),
            # CloserToEnemyBaseRewardFunction(),
        ])
        self.start_client()

        # computed properties
        # [num_planes_hp(5), num_planes_resources(5), num_planes_player(5),
        # num_planes_unit_type(z), num_planes_unit_action(6)]
        self.num_planes = [5, 5, 3, len(self.utt['unitTypes']) + 1, 6]
        self.observation_space = gym.spaces.Box(low=0.0,
                                                high=1.0,
                                                shape=(self.height, self.width,
                                                       sum(self.num_planes)),
                                                dtype=np.int32)
        self.action_space = gym.spaces.MultiDiscrete([
            self.height * self.width,
            6, 4, 4, 4, 4,
            len(self.utt['unitTypes']),
            7 * 7
        ])

    def start_client(self):
        from ts import JNIVecClient
        from ai.core import AI
        self.vec_client = JNIVecClient(
            self.num_envs,
            self.max_steps,
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_path,
            JArray(AI)([ai2(self.real_utt) for ai2 in self.ai2s]),
            self.real_utt
        )
        # get the unit type table
        self.utt = json.loads(str(self.vec_client.clients[0].sendUTT()))
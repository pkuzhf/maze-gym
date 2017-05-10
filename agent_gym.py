import numpy as np
import config, utils, copy

import gym
from gym import spaces
from rl.core import Processor
from gym.utils import seeding




class AGENT_GYM(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, ini_mazemap=None):

        self.action_space = spaces.Discrete(4)
        t = ()
        for i in range(config.Map.Height * config.Map.Width):
            t += (spaces.Discrete(4),)
        self.observation_space = spaces.Tuple(t)

        self.ini_mazemap = ini_mazemap

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        #utils.displayMap(self.ini_mazemap)
        [sx, sy, tx, ty] = utils.findSourceAndTarget(self.ini_mazemap)
        self.source = np.array([sx, sy])
        self.target = np.array([tx, ty])
        self.mazemap = copy.deepcopy(self.ini_mazemap)
        return self.mazemap

    def _step(self, action):

        done = False
        reward = -1

        new_source = self.source + utils.dirs[action]

        if utils.inMap(new_source[0], new_source[1]):

            if utils.equalCellValue(self.mazemap, new_source[0], new_source[1], utils.Cell.Target):
                #reward = 1
                done = True
                utils.setCellValue(self.mazemap, self.source[0], self.source[1], utils.Cell.Empty)
                utils.setCellValue(self.mazemap, new_source[0], new_source[1], utils.Cell.Source)
                self.source = new_source
                #utils.displayMap(self.mazemap)

            if utils.equalCellValue(self.mazemap, new_source[0], new_source[1], utils.Cell.Empty):
                utils.setCellValue(self.mazemap, self.source[0], self.source[1], utils.Cell.Empty)
                utils.setCellValue(self.mazemap, new_source[0], new_source[1], utils.Cell.Source)
                self.source = new_source
                #utils.displayMap(self.mazemap)

        return self.mazemap, reward, done, {}


class ADVERSARIAL_AGENT_GYM(AGENT_GYM):

    def __init__(self, env_gym, rollout_policy):
        self.env_gym = env_gym
        self.rollout_policy = rollout_policy
        super(ADVERSARIAL_AGENT_GYM, self).__init__()

    def _reset(self):
        #print '\nreset adversarial_agent_gym'
        #np.random.seed(config.Game.Seed)
        #self.env_gym.seed(config.Game.Seed)
        while True:
            self.ini_mazemap = self.env_gym.rollout_env_map(policy=self.rollout_policy)
            if self.env_gym.isvalid_mazemap(self.ini_mazemap):
                break
        utils.displayMap(self.ini_mazemap)

        return super(ADVERSARIAL_AGENT_GYM, self)._reset()

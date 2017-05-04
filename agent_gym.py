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
        reward = -0.01

        new_source = self.source + utils.dirs[action]

        if utils.inMap(new_source[0], new_source[1]):

            if utils.equalCellValue(self.mazemap, new_source[0], new_source[1], utils.Cell.Target):
                utils.setCellValue(self.mazemap, self.source[0], self.source[1], utils.Cell.Empty)
                utils.setCellValue(self.mazemap, new_source[0], new_source[1], utils.Cell.Source)
                self.source = new_source
                #utils.displayMap(self.mazemap)
                reward = 1
                done = True

            if utils.equalCellValue(self.mazemap, new_source[0], new_source[1], utils.Cell.Empty):
                utils.setCellValue(self.mazemap, self.source[0], self.source[1], utils.Cell.Empty)
                utils.setCellValue(self.mazemap, new_source[0], new_source[1], utils.Cell.Source)
                self.source = new_source
                #utils.displayMap(self.mazemap)

        return self.mazemap, reward, done, {}


class ADVERSARIAL_AGENT_GYM(AGENT_GYM):

    def __init__(self, env_gym):
        self.env_gym = env_gym
        super(ADVERSARIAL_AGENT_GYM, self).__init__()

    def _reset(self):
        print '\nreset adversarial_agent_gym'
        np.random.seed(config.Game.Seed)
        self.env_gym.seed(config.Game.Seed)
        self.ini_mazemap = self.env_gym.get_env_map()
        utils.displayMap(self.ini_mazemap)
        np.random.seed(None)

        return super(ADVERSARIAL_AGENT_GYM, self)._reset()

'''
class strong_agent_gym(agent_gym):

    def _reset(self, mazemap = None):
        n = config.Map.Height
        m = config.Map.Width
        if mazemap == None:

            mazemap = []
            for i in range(n):
                mazemap.append([])
                for j in range(m):
                    mazemap[i].append([utils.Cell.Empty] * 4)
                    utils.setCellValue(mazemap, i, j, np.random.binomial(utils.Cell.Wall, config.Map.WallDense))

            while True:
                sx = np.random.randint(n)
                sy = np.random.randint(m)
                if utils.equalCellValue(mazemap, sx, sy, utils.Cell.Empty):
                    utils.setCellValue(mazemap, sx, sy, utils.Cell.Source)
                    break

            f = open(config.StrongMazeEnv.EvaluateFile, 'r')
            distance = 1
            for line in f:
                [distance, score] = line.split()
                distance = int(distance)
                score = float(score)
                if score < config.StrongMazeEnv.ScoreLevel:
                    break
            f.close()

            while True:
                hasValidCell = False
                for i in range(n):
                    for j in range(m):
                        if utils.getDistance(sx, sy, i, j) == distance and utils.equalCellValue(mazemap, i, j, utils.Cell.Empty):
                            hasValidCell = True
                if hasValidCell:
                    break
                else:
                    distance += 1

            while True:
                tx = np.random.randint(n)
                ty = np.random.randint(m)
                if utils.getDistance(sx, sy, tx, ty) == distance and utils.equalCellValue(mazemap, tx, ty, utils.Cell.Empty):
                    utils.setCellValue(mazemap, tx, ty, utils.Cell.Target)
                    break
        else:
            [sx, sy, tx, ty] = findSourceAndTarget(mazemap)

        self.source = np.array([sx, sy])
        self.target = np.array([tx, ty])
        self.mazemap = np.array(mazemap)
        return self.mazemap
'''

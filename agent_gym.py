import numpy as np
import config, utils

import gym
from gym import spaces
from rl.core import Processor


def displayMap(mazemap):
    output = ''
    for i in range(len(mazemap)):
        for j in range(len(mazemap[i])):
            for k in range(len(mazemap[i][j])):
                if mazemap[i][j][k] == 1:
                    output += str(k)
        output += '\n'
    print output


def findSourceAndTarget(mazemap):
    for i in range(len(mazemap)):
        for j in range(len(mazemap[i])):
            if utils.getCellValue(mazemap, i, j) == config.Cell.Source:
                sx = i
                sy = j
            if utils.getCellValue(mazemap, i, j) == config.Cell.Target:
                tx = i
                ty = j
    return [sx, sy, tx, ty]


class agent_gym(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, mazemap=None):

        self.action_space = spaces.Discrete(4)
        t = ()
        for i in range(config.Map.Height * config.Map.Width):
            t += (spaces.Discrete(4),)
        self.observation_space = spaces.Tuple(t)

        self._seed()

        self._reset(mazemap)

    def _reset(self, mazemap=None):
        n = config.Map.Height
        m = config.Map.Width
        if mazemap == None:
            mazemap = []
            for i in range(n):
                mazemap.append([])
                for j in range(m):
                    mazemap[i].append(np.zeros(4))
                    utils.setCellValue(mazemap, i, j, np.random.binomial(config.Cell.Wall, config.Map.WallDense))
            while True:
                sx = np.random.randint(n)
                sy = np.random.randint(m)
                if utils.getCellValue(mazemap, sx, sy) == config.Cell.Empty:
                    utils.setCellValue(mazemap, sx, sy, config.Cell.Source)
                    break
            while True:
                tx = np.random.randint(n)
                ty = np.random.randint(m)
                if utils.getCellValue(mazemap, tx, ty) == config.Cell.Empty:
                    utils.setCellValue(mazemap, tx, ty, config.Cell.Target)
                    break
        else:
            [sx, sy, tx, ty] = findSourceAndTarget(mazemap)
               
        self.source = np.array([sx, sy])
        self.target = np.array([tx, ty])
        self.mazemap = np.array(mazemap)
        return self.mazemap

    def _step(self, action):
        reward = 0
        new_source = self.source + utils.dirs[action]
        if not utils.inMap(new_source[0], new_source[1]):
            done = False
        else:
            cell = utils.getCellValue(self.mazemap, new_source[0], new_source[1])
            if cell == config.Cell.Target: 
                done = True
                reward = 1
            else:
                done = False
            utils.setCellValue(self.mazemap, self.source[0], self.source[1], config.Cell.Empty)
            utils.setCellValue(self.mazemap, new_source[0], new_source[1], config.Cell.Source)
        return self.mazemap, reward, done, {}


class adversarial_agent_gym(agent_gym):

    def __init__(self, env_generator):
        self.env_generator = env_generator
        super(adversarial_agent_gym, self).__init__()

    def _reset(self, mazemap = None):
        mazemap = self.env_generator.get_env_map()
        [sx, sy, tx, ty] = findSourceAndTarget(mazemap)
        self.source = np.array([sx, sy])
        self.target = np.array([tx, ty])
        self.mazemap = np.array(mazemap)
        return self.mazemap


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
                    mazemap[i].append([config.Cell.Empty] * 4)
                    utils.setCellValue(mazemap, i, j, np.random.binomial(config.Cell.Wall, config.Map.WallDense))

            while True:
                sx = np.random.randint(n)
                sy = np.random.randint(m)
                if utils.getCellValue(mazemap, sx, sy) == config.Cell.Empty:
                    utils.setCellValue(mazemap, sx, sy, config.Cell.Source)
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
                        if utils.getDistance(sx, sy, i, j) == distance and utils.getCellValue(mazemap, i, j) == config.Cell.Empty:
                            hasValidCell = True
                if hasValidCell:
                    break
                else:
                    distance += 1

            while True:
                tx = np.random.randint(n)
                ty = np.random.randint(m)
                if utils.getDistance(sx, sy, tx, ty) == distance and utils.getCellValue(mazemap, tx, ty) == config.Cell.Empty:
                    utils.setCellValue(mazemap, tx, ty, config.Cell.Target)
                    break
        else:
            [sx, sy, tx, ty] = findSourceAndTarget(mazemap)

        self.source = np.array([sx, sy])
        self.target = np.array([tx, ty])
        self.mazemap = np.array(mazemap)
        return self.mazemap
'''

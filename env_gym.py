import copy
import config, utils
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from agent_gym import AGENT_GYM


class ENV_GYM(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, agent_net, env_net):
        self.env_net = env_net
        self.agent_net = agent_net

        self.action_space = spaces.Discrete(config.Map.Height * config.Map.Width)

        t = ()
        for i in range(config.Map.Height * config.Map.Width):
            t += (spaces.Discrete(4),)
        self.observation_space = spaces.Tuple(t)

        self._seed()

        self.env = None
        self.agent = None

    def _reset(self):
        self.gamestep = 0
        self.mazemap = utils.initMazeMap()
        return self.mazemap

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)

        done = False
        [x, y] = [action / config.Map.Width, action % config.Map.Width]
        if utils.equalCellValue(self.mazemap, x, y, utils.Cell.Empty):
            if self.gamestep == 0:
                utils.setCellValue(self.mazemap, x, y, utils.Cell.Source)
            elif self.gamestep == 1:
                utils.setCellValue(self.mazemap, x, y, utils.Cell.Target)
            else:
                utils.setCellValue(self.mazemap, x, y, utils.Cell.Wall)
            self.gamestep += 1
        elif self.gamestep >= 2:
            done = True

        mazemap = self.rollout_env_map(copy.deepcopy(self.mazemap))
        reward = self._get_reward_from_agent(mazemap)
        # print 'roll-out map'
        # utils.displayMap(mazemap)

        print ['gamestep', self.gamestep, 'reward', reward]
        utils.displayMap(mazemap)
        utils.displayMap(self.mazemap)
        #if done:
        #    utils.displayMap(self.mazemap)

        return self.mazemap, reward, done, {}

    def _get_reward_from_agent(self, mazemap):

        if not self.isvalid_mazemap(mazemap):
            return -100
        else:
            return utils.not_empty_count(mazemap)


        agent_gym = AGENT_GYM(mazemap)
        agent_gym.reset()

        gamestep = 0
        reward_episode = 0
        while gamestep < config.Game.MaxGameStep:
            gamestep += 1
            action = self.agent.forward(mazemap)
            obs, reward, done, info = agent_gym.step(action)
            reward_episode += reward
            if done:
                break

        return -reward_episode

    def rollout_env_map(self, mazemap=None):

        if mazemap is None:
            mazemap = utils.initMazeMap()

        not_empty_count = utils.not_empty_count(mazemap)

        while True:
            action = self.env.forward(mazemap)
            [x, y] = [action / config.Map.Width, action % config.Map.Width]
            if utils.equalCellValue(mazemap, x, y, utils.Cell.Empty):
                if not_empty_count == 0:
                    utils.setCellValue(mazemap, x, y, utils.Cell.Source)
                elif not_empty_count == 1:
                    utils.setCellValue(mazemap, x, y, utils.Cell.Target)
                else:
                    utils.setCellValue(mazemap, x, y, utils.Cell.Wall)
                not_empty_count += 1
            else: #if not_empty_count >= 2: #replaced with valid check
                break

        return mazemap

    def isvalid_mazemap(self, mazemap):

        [sx, sy, tx, ty] = utils.findSourceAndTarget(mazemap)
        if sx == 0 or sy == 0 or tx == 0 or ty == 0:
            return False

        from collections import deque
        queue = deque()
        queue.append([sx,sy])
        visited = np.zeros([config.Map.Height, config.Map.Width], dtype=np.int)

        while len(queue):
            [cx, cy] = queue.popleft()
            visited[cx][cy] = 1

            for k in range(len(utils.dirs)):
                [nx, ny] = [cx, cy] + utils.dirs[k]
                if not utils.inMap(nx, ny) or visited[nx][ny]:
                    continue
                if utils.equalCellValue(mazemap, nx, ny, utils.Cell.Empty):
                    queue.append([nx, ny])
                if nx == tx and ny == ty:
                    return True

        print 'Invalid Map'
        utils.displayMap(mazemap)

        return False
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

        agent_rewards = []
        for _ in range(config.Generator.RewardSampleN):
            if done:
                mazemap = copy.deepcopy(self.mazemap)
            else:
                mazemap = self.get_env_map(copy.deepcopy(self.mazemap))
            agent_rewards.append(self._get_reward_from_agent(mazemap))

        reward_mean, reward_std = np.mean(agent_rewards), np.std(agent_rewards)
        reward = -reward_mean

        #print ['gamestep', self.gamestep]
        #utils.displayMap(self.mazemap)
        #print [reward, reward_mean, reward_std]
        if done:
            utils.displayMap(self.mazemap)

        return self.mazemap, reward, done, {}

    def _get_reward_from_agent(self, mazemap):
        #print 'roll-out map'
        #utils.displayMap(mazemap)

        gamestep = 0
        reward_episode = 0
        agent_gym = AGENT_GYM(mazemap)
        while gamestep < config.Game.MaxGameStep:
            gamestep += 1
            action = self.agent.forward(mazemap)
            obs, reward, done, info = agent_gym.step(action)
            reward_episode += reward
            if done:
                break

        return reward_episode

    def get_env_map(self, mazemap=None):

        if mazemap==None:
            mazemap = utils.initMazeMap()

        not_empty_count = 0
        for i in range(config.Map.Height):
            for j in range(config.Map.Width):
                if utils.nequalCellValue(mazemap, i, j, utils.Cell.Empty):
                    not_empty_count += 1

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
            elif not_empty_count >= 2:
                break

        return mazemap
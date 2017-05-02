import copy
import config, utils
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from agent_gym import agent_gym


class env_gym(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, agent_net, env_gen):
        self.agent_net = agent_net
        self.env_gen = env_gen

        self.action_space = spaces.Discrete(config.Map.Height * config.Map.Width)

        t = ()
        for i in range(config.Map.Height * config.Map.Width):
            t += (spaces.Discrete(4),)
        self.observation_space = spaces.Tuple(t)

        self._seed()
        self._reset()

    def _reset(self):
        self.gamestep = 0
        self.mazemap = utils.initMazeMap()
        return self._get_obs()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        print ['gamestep', self.gamestep]
        utils.displayMap(self.mazemap)

        done = False
        [x, y] = [action / config.Map.Width, action % config.Map.Width]
        if utils.getCellValue(self.mazemap, x, y) == config.Cell.Empty:
            if self.gamestep == 0:
                utils.setCellValue(self.mazemap, x, y, config.Cell.Source)
            elif self.gamestep == 1:
                utils.setCellValue(self.mazemap, x, y, config.Cell.Target)
            else:
                utils.setCellValue(self.mazemap, x, y, config.Cell.Wall)
            self.gamestep += 1
        elif self.gamestep >= 2:
            done = True

        agent_rewards = []
        for _ in range(config.Generator.RewardSampleN):
            if done:
                mazemap = copy.deepcopy(self.mazemap)
            else:
                mazemap = self.env_gen.get_env_map(copy.deepcopy(self.mazemap))
            agent_rewards.append(self._get_reward_from_agent(mazemap))

        reward_mean, reward_std = np.mean(agent_rewards), np.std(agent_rewards)
        reward = reward_mean + reward_std
        print [reward, reward_mean, reward_std]

        return self._get_obs(), reward, done, {}

    def _get_reward_from_agent(self, mazemap):
        #print 'roll-out map'
        #utils.displayMap(mazemap)

        env = agent_gym(mazemap)
        #obs = env.reset()
        obs = mazemap
        reward_episode = 0
        gamestep = 0
        while True:
            if gamestep == config.GeneratorEnv.MaxGameStep:
                break
            gamestep += 1
            prob_n = self.agent_net.predict(np.array([[obs]]))
            action = utils.categoricalSample(prob_n, self.np_random)
            obs, reward, done, info = env.step(action)
            reward_episode += reward
            if done:
                break
        return reward_episode

    def _get_obs(self):
        return self.mazemap

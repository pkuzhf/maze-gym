import gym
from gym import spaces
from gym.utils import seeding
import config, utils
import copy
from gym.envs.toy_text.maze import MazeEnv
import numpy as np

class MazeGeneratorEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, agent_model, map_generator):
        self.agent_model = agent_model
        self.map_generator = map_generator

        self.action_space = spaces.Discrete(config.Map.Height * config.Map.Width + 1)

        t = ()
        for i in range(config.Map.Height * config.Map.Width):
            t += (spaces.Discrete(4),)
        self.observation_space = spaces.Tuple(t)

        self._seed()

        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        print ['self.gamestep', self.gamestep]
        print utils.displayMap(self.mazemap)
        if action == config.Map.Height * config.Map.Width or self.gamestep == config.Map.Height * config.Map.Width:
            if self.gamestep >= 2:
                done = True
                mazemap = copy.deepcopy(self.mazemap)
            else:
                done = False
        else:
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
        
        agent_rewards = []
        for _ in range(config.Generator.RewardSampleN):
            if not done:
                mazemap = self.map_generator.generate(copy.deepcopy(self.mazemap))
            agent_rewards.append(self._get_reward_from_agent(mazemap))
        reward = np.mean(agent_rewards) + np.std(agent_rewards)
        return self._get_obs(), reward, done, {}

    def _get_reward_from_agent(self, mazemap):
        print 'roll-out map'
        utils.displayMap(mazemap)
        env = MazeEnv(mazemap)
        obs = env.reset()
        reward_episode = 0
        gamestep = 0
        while True:
            if gamestep == config.GeneratorEnv.MaxGameStep:
                break
            gamestep += 1
            prob_n = self.agent_model.predict(np.array([[obs]]))
            action = utils.categoricalSample(prob_n, self.np_random)
            obs, reward, done, info = env.step(action)
            reward_episode += reward
            if done:
                break
        return reward_episode

    def _get_obs(self):
        return self.mazemap

    def _reset(self):
        self.gamestep = 0
        self.mazemap = utils.initMazeMap()
        return self._get_obs()
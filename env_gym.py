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

        self.action_space = spaces.Discrete(config.Map.Height*config.Map.Width)

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

    def _act(self, mazemap, action):

        done = False
        [x, y] = [action / config.Map.Width, action % config.Map.Width]
        if utils.equalCellValue(mazemap, x, y, utils.Cell.Empty):
            utils.setCellValue(mazemap, x, y, utils.Cell.Wall)
        else:
            done = True
            
        return done
        
    def _step(self, action):

        assert self.action_space.contains(action)
        done = self._act(self.mazemap, action)

        agent_rewards = []
        if done:
            mazemap = copy.deepcopy(self.mazemap)
            agent_rewards.append(self._get_reward_from_agent(mazemap))
        else:
            for _ in range(config.Generator.RolloutSampleN):
                mazemap = self.rollout_env_map(self.mazemap)
                agent_rewards.append(self._get_reward_from_agent(mazemap))
        reward_mean, reward_std = np.mean(agent_rewards), np.std(agent_rewards)
        reward = reward_mean

        self.gamestep += 1
        #print ['gamestep', self.gamestep, 'reward', reward]
        #utils.displayMap(self.mazemap)
        #utils.displayMap(mazemap)
        #if done:
        #    utils.displayMap(self.mazemap)

        return self.mazemap, reward, done, {}

    def _get_reward_from_agent(self, mazemap):

        #if not self.isvalid_mazemap(mazemap):
        #    return -1
        #else:
        return utils.nonempty_count(mazemap)-2

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
        else:
            mazemap = copy.deepcopy(mazemap)

        while True:
            action = self.env.forward(mazemap)
            done = self._act(mazemap, action)
            if done:
                break

        return mazemap

    def isvalid_mazemap(self, mazemap):

        [sx, sy, tx, ty] = utils.findSourceAndTarget(mazemap)
        if sx == -1 or sy == -1 or tx == -1 or ty == -1:
            print 'Invalid Map'
            utils.displayMap(mazemap)
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
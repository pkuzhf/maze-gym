import copy
import config, utils
import numpy as np
from collections import deque

import gym, os
from policy import *
from rl.agents.ddpg import DDPGAgent
from rl.agents.dqn import DQNAgent
from gym import spaces
from gym.utils import seeding
from agent_gym import AGENT_GYM


class ENV_GYM(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.action_space = spaces.Discrete(config.Map.Height*config.Map.Width+1)

        t = ()
        for i in range(config.Map.Height * config.Map.Width):
            t += (spaces.Discrete(4),)
        self.observation_space = spaces.Tuple(t)

        self._seed()

        self.env = None
        self.agent = None
        self.mask = None
        self.conflict_count = 0
        self.reward_his = deque(maxlen=10000)
        self.action_reward_his = [np.zeros(2) for _ in range(self.action_space.n)]

    def _reset(self):
        self.gamestep = 0
        self.conflict_count = 0
        self.mazemap = utils.initMazeMap()
        if type(self.env) == DQNAgent:
            if 'Masked' in type(self.env.policy).__name__:
                self.mask = self._getmask(self.mazemap)
                self.env.policy.mask = self.mask
        return self.mazemap

    def _getmask(self, mazemap):
        mask = np.zeros(self.action_space.n)
        for i in range(config.Map.Height):
            for j in range(config.Map.Width):
                if utils.nequalCellValue(mazemap, i, j, utils.Cell.Empty):
                    mask[i*config.Map.Width+j] = 1
        return mask

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _act(self, mazemap, action):

        done = (action == config.Map.Height * config.Map.Width)

        conflict = False

        if not done:

            [x, y] = [action / config.Map.Width, action % config.Map.Width]

            if utils.equalCellValue(mazemap, x, y, utils.Cell.Empty):
                utils.setCellValue(mazemap, x, y, utils.Cell.Wall)
            else:
                self.conflict_count += 1
                conflict = True
                done = True

            if type(self.env) == DQNAgent:
                if 'Masked' in type(self.env.policy).__name__:
                    self.env.policy.mask[action] = 1

        if done:
            for i in range(config.Map.Height):
                for j in range(config.Map.Width):
                    if utils.equalCellValue(mazemap, i, j, utils.Cell.Wall):
                        utils.setCellValue(mazemap, i, j, utils.Cell.Wall2)

        return done, conflict

    def _get_action(self, action_base_vec):

        if type(self.env) == DQNAgent:
            return action_base_vec
        else:
            #print action_base_vec
            action_base_vec -= np.max(action_base_vec)
            exp_values = np.exp(action_base_vec)
            probs = exp_values / np.sum(exp_values)
            action = np.random.choice(range(self.action_space.n), p=probs)
            #print action
            return action

    def _step(self, action):

        action = self._get_action(action)

        assert self.action_space.contains(action)
        done, conflict = self._act(self.mazemap, action)

        reward = 0
        if done:
            mazemap = copy.deepcopy(self.mazemap)
            reward = self._get_reward_from_agent(mazemap)
        #else:
        #    agent_rewards = []
        #    for _ in range(config.Generator.RolloutSampleN):
        #        mazemap = self.rollout_env_map(self.mazemap)
        #        agent_rewards.append(self._get_reward_from_agent(mazemap))
        #    reward_mean, reward_std = np.mean(agent_rewards), np.std(agent_rewards)
        #    reward = reward_mean

        self.gamestep += 1
        #if not conflict:
        #    print ['gamestep', self.gamestep, 'confilict', self.conflict_count, 'reward', reward]
        #    utils.displayMap(self.mazemap)
            #utils.displayMap(mazemap)
        if done:
            #state = self.env.memory.get_recent_state(self.mazemap)
            #q_values = self.env.compute_q_values(state)
            #print q_values
            #os.system("clear")
            print ['gamestep', self.gamestep, 'confilict', self.conflict_count, 'reward', reward, 'his_avg_reward', np.mean(self.reward_his)]
            #print self.action_reward_his
            utils.displayMap(self.mazemap)
            self.reward_his.append(reward)

        count = self.action_reward_his[action][0]
        self.action_reward_his[action][0] += 1
        self.action_reward_his[action][1] = (self.action_reward_his[action][1] * count + reward) / (count + 1)

        return self.mazemap, reward, done, {}

    def _get_reward_from_agent(self, mazemap):

        #if not self.isvalid_mazemap(mazemap):
        #    return -1
        #else:
        return utils.Wall2_count(mazemap) * 0.01

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

        if type(self.env) == DQNAgent:
            if 'Masked' in type(self.env.policy).__name__:
                mask = self._getmask(mazemap)
                self.env.policy.mask = mask

        while True:
            action = self._get_action(self.env.forward(mazemap))
            done, conflict = self._act(mazemap, action)
            if done:
                break

        if type(self.env) == DQNAgent:
            if type(self.env.policy) == MaskedBoltzmannQPolicy2:
                self.env.policy.mask = self.mask

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
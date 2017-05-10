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
        self.invalid_count = 0
        self.conflict_count = 0
        self.max_reward = -1e20
        self.reward_his = deque(maxlen=10000)
        self.action_reward_his = [np.zeros(2) for _ in range(self.action_space.n)]

    def _reset(self):
        self.gamestep = 0
        self.invalid_count = 0
        self.conflict_count = 0
        self.mazemap = utils.initMazeMap()
        self.mask = self._getmask(self.mazemap)
        self.env.policy.set_mask(self.mask)
        self.env.test_policy.set_mask(self.mask)
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

    def _update_mask(self, action, mask, mazemap, mask_invalid=True):

        mask[action] = 1

        if mask_invalid:
            for i in range(0, config.Map.Height):
                for j in range(0, config.Map.Width):
                    if utils.equalCellValue(mazemap, i, j, utils.Cell.Empty):
                        utils.setCellValue(mazemap, i, j, utils.Cell.Wall)
                        if not self.isvalid_mazemap(mazemap):
                            mask[i*config.Map.Width+j] = 1
                        utils.setCellValue(mazemap, i, j, utils.Cell.Empty)

    def _act(self, mazemap, action, mask):

        done = (action == config.Map.Height * config.Map.Width)

        invalid = False
        conflict = False

        if not done:

            x, y = action / config.Map.Width, action % config.Map.Width

            if utils.equalCellValue(mazemap, x, y, utils.Cell.Empty):
                utils.setCellValue(mazemap, x, y, utils.Cell.Wall)
                if not self.isvalid_mazemap(mazemap):
                    utils.setCellValue(mazemap, x, y, utils.Cell.Empty)
                    self.invalid_count += 1
                    invalid = True
                    done = True
            else:
                self.conflict_count += 1
                conflict = True
                done = True

            if mask is not None:
                self._update_mask(action, mask, mazemap)

        return done, conflict, invalid

    def _step(self, action):
        assert self.action_space.contains(action)

        done, conflict, invalid = self._act(self.mazemap, action, self.mask)

        if done:
            mazemap = copy.deepcopy(self.mazemap)
            reward = self._get_reward_from_agent(mazemap)
        else:
            reward = 0

        self.gamestep += 1
        if done:
            self.reward_his.append(reward)
            self.max_reward = max(self.max_reward, reward)
            print('gamestep', self.gamestep, 'conflict/invalid', '%d / %d' % (self.conflict_count, self.invalid_count), 'reward', '%0.2f / %0.2f' % (reward, self.max_reward), 'avg_r', '%0.2f' % np.mean(self.reward_his),
                  'minq', '%0.2f / %0.2f' % (self.env.policy.cur_minq, self.env.policy.minq), 'maxq', '%0.2f : %0.2f / %0.2f ' % (self.env.policy.cur_maxq-reward, self.env.policy.cur_maxq, self.env.policy.maxq),
                  'eps', '%0.2f / %0.2f)' % (self.env.policy.eps_forB, self.env.policy.eps_forC))
            utils.displayMap(self.mazemap)

        return self.mazemap, reward, done, {}

    def _get_reward_from_agent(self, mazemap):

        #return utils.Wall_count(mazemap)
        #return self.random_path(mazemap)
        #return self.shortest_path(mazemap)
        #return self.shortest_random_path(mazemap)
        #return self.rightdown_path(mazemap)
        #return self.rightdownupleft_path(mazemap)
        #return self.rightdown_random_path(mazemap)
        #return self.dfs_path(mazemap)

        agent_gym = AGENT_GYM(mazemap)
        agent_gym.reset()

        gamestep = 0
        reward_episode = 0
        while gamestep < config.Game.MaxGameStep:
            gamestep += 1
            action = self.agent.forward(agent_gym.mazemap)
            obs, reward, done, info = agent_gym.step(action)
            reward_episode += reward
            if done:
                break

        return -reward_episode

    def rollout_env_map(self, mazemap=None, policy=None): #mazemap and policy state might get change

        if mazemap is None:
            mazemap = utils.initMazeMap()

        if policy == None:
            policy = self.env.test_policy

        mask = self._getmask(mazemap)
        policy.set_mask(mask)

        while True:
            q_values = self.env.compute_q_values([mazemap])
            action = policy.select_action(q_values=q_values)
            done, conflict, invalid = self._act(mazemap, action, policy.mask)
            if done:
                break

        return mazemap

    def isvalid_mazemap(self, mazemap):

        [sx, sy, tx, ty] = utils.findSourceAndTarget(mazemap)
        if sx == -1 or sy == -1 or tx == -1 or ty == -1:
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

        #print 'Invalid Map'
        #utils.displayMap(mazemap)
        return False

    def shortest_path(self, mazemap):

        [sx, sy, tx, ty] = utils.findSourceAndTarget(mazemap)
        if sx == -1 or sy == -1 or tx == -1 or ty == -1:
            return -1

        from collections import deque
        queue = deque()
        queue.append([sx, sy])
        shortest_path = np.zeros([config.Map.Height, config.Map.Width], dtype=np.int) # zero for unvisited
        shortest_path[sx][sy] = 1

        #utils.displayMap(mazemap)

        while len(queue):
            [cx, cy] = queue.popleft()
            cur_path_len = shortest_path[cx][cy]

            for k in range(len(utils.dirs)):
                [nx, ny] = [cx, cy] + utils.dirs[k]
                if not utils.inMap(nx, ny):
                    continue
                if utils.equalCellValue(mazemap, nx, ny, utils.Cell.Empty) or utils.equalCellValue(mazemap, nx, ny, utils.Cell.Target):
                    if shortest_path[nx][ny] == 0 or shortest_path[nx][ny] > cur_path_len + 1:
                        queue.append([nx, ny])
                        shortest_path[nx][ny] = cur_path_len + 1

        #print('shortest_path:' + str(shortest_path[tx][ty]))

        return shortest_path[tx][ty]

    def shortest_random_path(self, mazemap):

        [sx, sy, tx, ty] = utils.findSourceAndTarget(mazemap)
        if sx == -1 or sy == -1 or tx == -1 or ty == -1:
            return -1

        from collections import deque
        queue = deque()
        queue.append([tx, ty])
        shortest_path = np.zeros([config.Map.Height, config.Map.Width], dtype=np.int) # zero for unvisited
        shortest_path[tx][ty] = 0

        #utils.displayMap(mazemap)

        while len(queue):
            [cx, cy] = queue.popleft()
            cur_path_len = shortest_path[cx][cy]

            for k in range(len(utils.dirs)):
                [nx, ny] = [cx, cy] + utils.dirs[k]
                if not utils.inMap(nx, ny):
                    continue
                if utils.equalCellValue(mazemap, nx, ny, utils.Cell.Empty) or utils.equalCellValue(mazemap, nx, ny, utils.Cell.Source):
                    if shortest_path[nx][ny] == 0 or shortest_path[nx][ny] > cur_path_len + 1:
                        queue.append([nx, ny])
                        shortest_path[nx][ny] = cur_path_len + 1

        # go optimal direction in probability $optimal_dir_prob
        step = 0
        max_step = 200
        optimal_dir_prob = 0.8
        invalid_distance = config.Map.Height * config.Map.Width
        while (sx != tx or sy != ty) and step < max_step:
            distance_dirs = []
            valid_dir_n = 0
            for i in range(len(utils.dirs)):
                dx = sx + utils.dirs[i][0]
                dy = sy + utils.dirs[i][1]
                if utils.inMap(dx, dy) and not utils.equalCellValue(mazemap, dx, dy, utils.Cell.Wall):
                    distance_dirs.append(shortest_path[dx][dy])
                    valid_dir_n += 1
                else:
                    distance_dirs.append(invalid_distance)
            prob_dirs = []
            for i in range(len(utils.dirs)):
                if i == np.argmin(distance_dirs):
                    prob_dirs.append(optimal_dir_prob)
                elif distance_dirs[i] != invalid_distance:
                    prob_dirs.append((1 - optimal_dir_prob) / (valid_dir_n - 1))
                else:
                    prob_dirs.append(0.)

            selected_dir = np.argmax(np.random.multinomial(1, prob_dirs))
            sx += utils.dirs[selected_dir][0]
            sy += utils.dirs[selected_dir][1]
            step += 1
        return step

    def random_path(self, mazemap):
        [sx, sy, tx, ty] = utils.findSourceAndTarget(mazemap)
        if sx == -1 or sy == -1 or tx == -1 or ty == -1:
            return -1
        step = 0
        max_step = 200
        while (sx != tx or sy != ty) and step < max_step:
            valid_dirs = []
            for i in range(len(utils.dirs)):
                dx = sx + utils.dirs[i][0]
                dy = sy + utils.dirs[i][1]
                if utils.inMap(dx, dy) and not utils.equalCellValue(mazemap, dx, dy, utils.Cell.Wall):
                    valid_dirs.append(i)
            selected_dir = valid_dirs[np.random.randint(len(valid_dirs))]
            sx += utils.dirs[selected_dir][0]
            sy += utils.dirs[selected_dir][1]
            step += 1
        return step

    def rightdown_path(self, mazemap):
        [sx, sy, tx, ty] = utils.findSourceAndTarget(mazemap)
        if sx == -1 or sy == -1 or tx == -1 or ty == -1:
            return -1
        step = 0
        max_step = 200
        while (sx != tx or sy != ty) and step < max_step:
            # right
            dx = sx + utils.dirs[0][0]
            dy = sy + utils.dirs[0][1]     
            if utils.inMap(dx, dy) and not utils.equalCellValue(mazemap, dx, dy, utils.Cell.Wall):
                sx = dx
                sy = dy
            else:
                # down
                dx = sx + utils.dirs[1][0]
                dy = sy + utils.dirs[1][1]     
                if utils.inMap(dx, dy) and not utils.equalCellValue(mazemap, dx, dy, utils.Cell.Wall):
                    sx = dx
                    sy = dy
            step += 1
        return step

    def rightdownupleft_path(self, mazemap):
        [sx, sy, tx, ty] = utils.findSourceAndTarget(mazemap)
        if sx == -1 or sy == -1 or tx == -1 or ty == -1:
            return -1
        step = 0
        max_step = 200
        while (sx != tx or sy != ty) and step < max_step:
            # deterministic order: right, down, up, left
            for i in range(len(utils.dirs)):
                dx = sx + utils.dirs[i][0]
                dy = sy + utils.dirs[i][1]     
                if utils.inMap(dx, dy) and not utils.equalCellValue(mazemap, dx, dy, utils.Cell.Wall):
                    sx = dx
                    sy = dy
                    break
            step += 1
        return step

    def rightdown_random_path(self, mazemap):
        [sx, sy, tx, ty] = utils.findSourceAndTarget(mazemap)
        if sx == -1 or sy == -1 or tx == -1 or ty == -1:
            return -1
        step = 0
        max_step = 200
        while (sx != tx or sy != ty) and step < max_step:
            while True:
                # right 0.4, down 0.4, up 0.1, left 0.1
                selected_dir = np.argmax(np.random.multinomial(1, [0.4, 0.4, 0.1, 0.1]))
                dx = sx + utils.dirs[selected_dir][0]
                dy = sy + utils.dirs[selected_dir][1]     
                if utils.inMap(dx, dy) and not utils.equalCellValue(mazemap, dx, dy, utils.Cell.Wall):
                    sx = dx
                    sy = dy
                    break
            step += 1
        return step        

    def dfs_path(self, mazemap):
        [sx, sy, tx, ty] = utils.findSourceAndTarget(mazemap)
        if sx == -1 or sy == -1 or tx == -1 or ty == -1:
            return -1
        
        # explore in a dfs way until find target
        stack = [[sx, sy]]
        step = 0
        visited = np.zeros([config.Map.Height, config.Map.Width], dtype=np.int) # zero for unvisited
        visited[sx][sy] = 1

        while len(stack) > 0:
            [x, y] = stack[-1]
            if x == tx and y == ty:
                break
            expended = False
            for i in range(len(utils.dirs)):
                dx = x + utils.dirs[i][0]
                dy = y + utils.dirs[i][1]
                if utils.inMap(dx, dy) and not utils.equalCellValue(mazemap, dx, dy, utils.Cell.Wall) and visited[dx][dy] == 0:
                    expended = True
                    visited[dx][dy] = 1
                    stack.append([dx, dy])
                    step += 1
                    break
            if not expended:
                stack.pop()
                step += 1

        return step
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
from collections import deque


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
        self.gamestep = 0
        self.invalid_count = 0
        self.conflict_count = 0
        self.max_reward = -1e20
        self.reward_his = deque(maxlen=1000)

        self.used_agent = False

    def _reset(self):
        self.gamestep = 0
        self.invalid_count = 0
        self.conflict_count = 0
        self.mazemap = utils.initMazeMap()

        if 'Masked' in type(self.env.policy).__name__ or 'Masked' in type(self.env.test_policy).__name__:
            self.mask = self._getmask(self.mazemap)
            self.env.policy.set_mask(self.mask)
            self.env.test_policy.set_mask(self.mask)
        return self.mazemap

    def _getmask(self, mazemap):
        mask = np.zeros(self.action_space.n)
        for i in range(config.Map.Height):
            for j in range(config.Map.Width):
                if not mazemap[i, j, utils.Cell.Empty]:
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
                    if mazemap[i, j, utils.Cell.Empty]:
                        mazemap[i, j] = utils.Cell.WallV
                        if not self.isvalid_mazemap(mazemap):
                            mask[i*config.Map.Width+j] = 1
                        mazemap[i, j] = utils.Cell.EmptyV

    def _act(self, mazemap, action, mask):

        done = (action == config.Map.Height * config.Map.Width)

        invalid = False
        conflict = False

        if not done:

            x, y = action / config.Map.Width, action % config.Map.Width

            if mazemap[x, y, utils.Cell.Empty]:
                mazemap[x, y] =  utils.Cell.WallV
                if not self.isvalid_mazemap(mazemap):
                    mazemap[x, y] = utils.Cell.EmptyV
                    self.invalid_count += 1
                    invalid = True
                    done = True
            else:
                self.conflict_count += 1
                conflict = True
                done = True

            if mask is not None and not done:
                self._update_mask(action, mask, mazemap)

        return done, conflict, invalid

    def _step(self, action):
        assert self.action_space.contains(action)

        done, conflict, invalid = self._act(self.mazemap, action, self.mask)

        if done:

            mazemap = copy.deepcopy(self.mazemap)
            reward = self._get_reward_from_agent(mazemap)

            if self.conflict_count or self.invalid_count:
                print(
                'env_step', self.gamestep, 'conflict/invalid', '%d / %d' % (self.conflict_count, self.invalid_count))

            utils.displayMap(self.mazemap)

            if self.used_agent:
                print('agent rewards: ' + utils.string_values(self.agent.reward_his) + '   agent qvalues: ' + utils.string_values(self.agent.q_values))

            # self.reward_his.append(reward)
            # self.max_reward = max(self.max_reward, reward)
            # print('env_step', self.gamestep, 'conflict/invalid', '%d / %d' % (self.conflict_count, self.invalid_count), 'reward', '%0.2f / %0.2f' % (reward, self.max_reward), 'avg_r', '%0.2f' % np.mean(self.reward_his),
            #      'minq', '%0.2f / %0.2f' % (self.qlogger.cur_minq, self.qlogger.minq), 'maxq', '%0.2f : %0.2f / %0.2f ' % (self.qlogger.cur_maxq-reward, self.qlogger.cur_maxq, self.qlogger.maxq),
            #      'eps', '%0.2f / %0.2f' % (self.env.policy.eps_forB, self.env.policy.eps_forC))

        else:
            reward = 0

        self.gamestep += 1

        return self.mazemap, reward, done, {}

    def _get_reward_from_agent(self, mazemap):

        # return self.Wall_count(mazemap)
        # return self.random_path(mazemap)
        # return self.shortest_random_path(mazemap)
        # return self.rightdown_path(mazemap)
        # return self.rightdownupleft_path(mazemap)
        # return self.rightdown_random_path(mazemap)
        #return self.shortest_path(mazemap)

        if 'dfs' in config.Game.Type:
            return self.dfs_path(mazemap)
        elif 'right_hand' in config.Game.Type:
            return self.right_hand_path(mazemap)
        elif 'shortest' in config.Game.Type:
            return self.shortest_path(mazemap)
        else:

            if 'dqn' not in config.Game.Type and 'default' not in config.Game.Type:
                print('taskname should be dfs, right_hand, shortest, dqn or dqn5')
                assert False

            self.used_agent = True

            agent_gym = AGENT_GYM(mazemap)
            agent_gym.agent = self.agent
            agent_gym.reset()

            fit_this_map = True
            if fit_this_map:
                self.agent.max_reward = -1e20
                self.agent.reward_his.clear()
                self.agent.memory.__init__(config.Training.BufferSize, window_length=1)
                # we do not reset the agent network, to accelerate the training.
                while True:
                    self.agent.fit(agent_gym, nb_episodes=10, min_steps=100+self.agent.nb_steps_warmup, nb_max_episode_steps=config.Game.MaxGameStep, visualize=False, verbose=0)
                    if np.min(self.agent.reward_his) != -config.Game.MaxGameStep:
                        break
                    else:
                        print('agent rewards: ' + utils.string_values(self.agent.reward_his) + '   agent qvalues: ' + utils.string_values(self.agent.q_values))
                        self.agent.reward_his.clear()
                        np.random.seed(None)
                if config.Game.AgentAction == 4:
                    return -self.agent.max_reward
                else: #return np.mean(self.agent.reward_his[:-10])
                    self.agent.test_reward_his.clear()
                    self.agent.test(agent_gym, nb_episodes=10, nb_max_episode_steps=config.Game.MaxGameStep, visualize=False, verbose=0)
                    return -np.mean(self.agent.test_reward_his)
            else:
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

        assert False
        return 0

    def rollout_env_map(self, mazemap=None, policy=None): #mazemap and policy state might get change

        if mazemap is None:
            mazemap = utils.initMazeMap()

        if policy is None:
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
                if mazemap[nx, ny, utils.Cell.Empty]:
                    queue.append([nx, ny])
                if nx == tx and ny == ty:
                    return True

        return False

    @staticmethod
    def shortest_path(mazemap):

        [sx, sy, tx, ty] = utils.findSourceAndTarget(mazemap)
        if sx == -1 or sy == -1 or tx == -1 or ty == -1:
            return -1

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
                if mazemap[nx, ny, utils.Cell.Empty] or mazemap[nx, ny, utils.Cell.Target]:
                    if shortest_path[nx][ny] == 0 or shortest_path[nx][ny] > cur_path_len + 1:
                        queue.append([nx, ny])
                        shortest_path[nx][ny] = cur_path_len + 1

        #print('shortest_path:' + str(shortest_path[tx][ty]))

        #if shortest_path[tx][ty]==11:
        #    utils.displayMap(mazemap)
        #    print('error')

        return shortest_path[tx][ty]-1

    def right_hand_path(self, mazemap):

        [sx, sy, tx, ty] = utils.findSourceAndTarget(mazemap)
        if sx == -1 or sy == -1 or tx == -1 or ty == -1:
            return -1

        mazemap[sx, sy] = utils.Cell.EmptyV

        count = 0
        cx, cy = sx, sy
        path = []

        cur_dir = 0
        dirs = np.array([[1, 0], [0, -1], [-1, 0], [0, 1]])
        p = [1, 0, 3, 2]
        while cx != tx or cy != ty:
            for i in p:
                next_dir = (cur_dir + i) % 4
                nx, ny = [cx, cy] + dirs[next_dir]
                if utils.inMap(nx, ny):
                    if mazemap[nx,ny,utils.Cell.Empty] or mazemap[nx,ny,utils.Cell.Target] :
                        cx, cy = nx, ny
                        cur_dir = next_dir
                        break
            count += 1
            path.append([cx, cy])

        mazemap[sx, sy] = utils.Cell.SourceV

        print(count, path)

        return count


    def shortest_random_path(self, mazemap):

        [sx, sy, tx, ty] = utils.findSourceAndTarget(mazemap)
        if sx == -1 or sy == -1 or tx == -1 or ty == -1:
            return -1

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
                if mazemap[nx, ny, utils.Cell.Empty] or mazemap[nx, ny, utils.Cell.Source]:
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
                if utils.inMap(dx, dy) and not mazemap[dx, dy, utils.Cell.Wall]:
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
                if utils.inMap(dx, dy) and not mazemap[dx, dy, utils.Cell.Wall]:
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
            if utils.inMap(dx, dy) and not mazemap[dx, dy, utils.Cell.Wall]:
                sx = dx
                sy = dy
            else:
                # down
                dx = sx + utils.dirs[1][0]
                dy = sy + utils.dirs[1][1]     
                if utils.inMap(dx, dy) and not mazemap[dx, dy, utils.Cell.Wall]:
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
                if utils.inMap(dx, dy) and not mazemap[dx, dy, utils.Cell.Wall]:
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
                if utils.inMap(dx, dy) and not mazemap[dx, dy, utils.Cell.Wall]:
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
                if utils.inMap(dx, dy) and not mazemap[dx, dy, utils.Cell.Wall] and visited[dx][dy] == 0:
                    expended = True
                    visited[dx][dy] = 1
                    stack.append([dx, dy])
                    step += 1
                    break
            if not expended:
                stack.pop()
                step += 1

        return step

    def Wall_count(self, mazemap):
        Wall_count = 0
        for i in range(0, config.Map.Height):
            for j in range(0, config.Map.Width):
                if mazemap[i, j, utils.Cell.Wall] == 1:
                    Wall_count += 1
        return Wall_count
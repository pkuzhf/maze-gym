import numpy as np
import sys
from six import StringIO
import copy

from gym import spaces, utils
from gym.envs.toy_text import discrete
from rl.core import Processor

dirs = [[0, 1], [1, 0], [-1, 0], [0, -1]]

class MazeEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):

        [nS, nA, P, isd] = self.initMazeMap()

        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def initMazeMap(self):
        n = 8
        m = 8

        mazemap = []
        for i in range(n):
            mazemap.append([])
            for j in range(m):
                mazemap[i].append(np.random.randint(2))
        sx = np.random.randint(n)
        sy = np.random.randint(m)
        tx = np.random.randint(n)
        ty = np.random.randint(m)
        while tx == sx and ty == sy:
            tx = np.random.randint(n)
            ty = np.random.randint(m)
            
        mazemap[sx][sy] = 0
        mazemap[tx][ty] = 3
        #print np.array(mazemap)

        self.mazemap = mazemap
        self.sx = sx
        self.sy = sy
        self.m = m

        nS = n * m 
        nA = len(dirs)        
        isd = np.zeros(nS)
        isd[self.encode(sx, sy, m)] = 1
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        for si in range(n):
            for sj in range(m):
                if mazemap[si][sj] == 1:
                    continue
                    if tx == si and ty == sj:
                        continue
                state = self.encode(si, sj, m)
                for a in range(len(dirs)):
                    dx = si + dirs[a][0]
                    dy = sj + dirs[a][1]
                    if dx == tx and dy == ty:
                        reward = 1
                        done = True
                    else:
                        reward = 0
                        done = False
                    if dx >= 0 and dx < n and dy >= 0 and dy < m and mazemap[dx][dy] == 0:
                        newstate = self.encode(dx, dy, m)
                    else:
                        newstate = self.encode(si, sj, m)
                    P[state][a].append((1.0, newstate, reward, done))
        return [nS, nA, P, isd]

    def encode(self, sx, sy, m):
        return sx * m + sy

    def decode(self, i, m):
        return [i / m, i % m]

    def genObservation(self, mazemap, state):
        n = len(mazemap)
        m = len(mazemap[0])
        [x, y] = self.decode(state, m)
        mazemap[x][y] = 2
        for i in range(n):
            for j in range(m):
                mazemap[i][j] = [mazemap[i][j]]
        #print['mazemap', mazemap]
        #print['self.mazemap', self.mazemap]
        return np.array(mazemap)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        [x, y] = self.decode(self.s, self.m)
        mazemap = self.mazemap
        mazemap[x][y] = 2
        #outfile.write('\n'.join([''.join(str(ele) for ele in row) for row in mazemap]) + '\n')
        mazemap[x][y] = 0

        # No need to return anything for human
        if mode != 'human':
            return outfile

    def _reset(self):
        [nS, nA, P, isd] = self.initMazeMap()
        self.P = P
        self.isd = isd
        self.lastaction=None # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self._seed()        
        self.s = discrete.DiscreteEnv._reset(self)
        return self.genObservation(copy.deepcopy(self.mazemap), self.s)

    def _step(self, action):
        [s, r, d, p] = discrete.DiscreteEnv._step(self, action)
        return [self.genObservation(copy.deepcopy(self.mazemap), s), r, d, p]


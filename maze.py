import numpy as np
import sys
from six import StringIO
import copy

from gym import spaces, utils
from gym.envs.toy_text import discrete
from rl.core import Processor

# right, down, up, left
dirs = [[0, 1], [1, 0], [-1, 0], [0, -1]]

# gen_method includes 'unification', 'manully_guided'
#gen_method = 'unification'
gen_method = 'manully_guided'

evaluate_log_file = '/home/zhf/drive200g/openaigym/code/evaluate.txt'

def inboard(x, y, n, m):
    return x >= 0 and x < n and y >= 0 and y < m

def setMapValue(mazemap, x, y, value):
    for i in range(len(mazemap[x][y])):
        mazemap[x][y][i] = 0
    mazemap[x][y][value] = 1

def getMapValue(mazemap, x, y):
    for i in range(len(mazemap[x][y])):
        if mazemap[x][y][i] == 1:
            return i

def displayMap(mazemap):
    output = ''
    for i in range(len(mazemap)):
        for j in range(len(mazemap[i])):
            for k in range(len(mazemap[i][j])):
                if mazemap[i][j][k] == 1:
                    output += str(k)
        output += '\n'
    print output

def getDistance(sx, sy, tx, ty):
    return abs(sx - tx) + abs(sy - ty)

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
                mazemap[i].append(np.zeros(4))
                setMapValue(mazemap, i, j, np.random.binomial(1, 0))
        while True:
            sx = np.random.randint(n)
            sy = np.random.randint(m)
            if getMapValue(mazemap, sx, sy) == 0:
                setMapValue(mazemap, sx, sy, 2)
                break

        if gen_method == 'unification':
            while True:
                tx = np.random.randint(n)
                ty = np.random.randint(m)
                if getMapValue(mazemap, tx, ty) == 0:
                    setMapValue(mazemap, tx, ty, 3)
                    break

        if gen_method == 'manully_guided':
            f = open(evaluate_log_file, 'r')
            distance = 1
            for line in f:
                [distance, score] = line.split()
                distance = int(distance)
                score = float(score)
                if score < 0.8:
                    break
            f.close()

            while True:
                hasValidCell = False
                for i in range(n):
                    for j in range(m):
                        if getDistance(sx, sy, i, j) == distance and getMapValue(mazemap, i, j) == 0:
                            hasValidCell = True
                if hasValidCell:
                    break
                else:
                    distance += 1
            while True:
                tx = np.random.randint(n)
                ty = np.random.randint(m)
                if getDistance(sx, sy, tx, ty) == distance and getMapValue(mazemap, tx, ty) == 0:
                    setMapValue(mazemap, tx, ty, 3)
                    break
        displayMap(mazemap)    
        setMapValue(mazemap, sx, sy, 0)
        

        self.mazemap = mazemap
        self.sx = sx
        self.sy = sy
        self.n = n
        self.m = m

        nS = n * m 
        nA = len(dirs)        
        isd = np.zeros(nS)
        isd[self.encode(sx, sy, m)] = 1
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        for si in range(n):
            for sj in range(m):
                if getMapValue(mazemap, si, sj) == 1:
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
                    if inboard(dx, dy, n, m) and getMapValue(mazemap, dx, dy) == 0:
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
        n = self.n
        m = self.m
        [x, y] = self.decode(state, m)
        
        setMapValue(mazemap, x, y, 2)
        
        #print['mazemap', mazemap]
        #print['self.mazemap', self.mazemap]
        return np.array(mazemap)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        [x, y] = self.decode(self.s, self.m)
        mazemap = self.mazemap
        setMapValue(mazemap, x, y, 2)
        #outfile.write(str(np.array(mazemap)) + '\n')
        setMapValue(mazemap, x, y, 0)

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
        #setMapValue(self.mazemap, self.sx, self.sy, 2)
        #print np.array(self.mazemap)
        #setMapValue(self.mazemap, self.sx, self.sy, 0)
        return self.genObservation(copy.deepcopy(self.mazemap), self.s)

    def _step(self, action):
        [s, r, d, p] = discrete.DiscreteEnv._step(self, action)
        return [self.genObservation(copy.deepcopy(self.mazemap), s), r, d, p]


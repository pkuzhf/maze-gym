import numpy as np
import sys
from six import StringIO

from gym import spaces, utils
from gym.envs.toy_text import discrete

mazemap = [
    "s0000000",
    "01000000",
    "00100000",
    "00010000",
    "00001000",
    "00000100",
    "00000010",
    "0000000t",
]

dirs = [[0, 1], [1, 0], [-1, 0], [0, -1]]

class MazeEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):

        n = len(mazemap)
        m = len(mazemap[0])
        sx = -1
        sy = -1
        for i in range(n):
            for j in range(m):
                if mazemap[i][j] == 's':
                    sx = i
                    sy = j
                    mazemap[i][j] = '0'
        self.mazemap = mazemap
        self.sx = sx
        self.sy = sy

        nS = n * m 
        nA = len(dirs)        
        isd = np.zeros(nS)
        isd[self.encode(sx, sy)]

        for si in range(n):
            for sj in range(m):
                if mazemap[si][sj] == '1':
                    continue
                    if tx == si and ty == sj:
                        continue
                for a in range(dirs):
                    dx = si + dirs[a][0]
                    dy = sj + dirs[a][1]
                    if dx == tx and dy == ty:
                        reward = 1
                        done = True
                    else:
                        reward = 0
                        done = False
                    if mazemap[dx][dy] == '0':
                        newstate = self.encode(dx, dy)
                    else:
                        newstate = self.encode(si, sj)
                    P[state][a].append((1.0, newstate, reward, done))
        
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def encode(self, sx, sy, m):
        return sx * m + sy

    def decode(self, i, m):
        return [i / m, i % m]

    def makeMap(self, mazemap, sx, sy):
        mazemap = mazemap.copy()
        mazemap[sx][sy] = 's'
        return mazemap

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        mazemap = self.makeMap(self.mazemap, self.sx, self.sy)
        outfile.write('\n'.join(mazemap) + '\n')

        # No need to return anything for human
        if mode != 'human':
            return outfile


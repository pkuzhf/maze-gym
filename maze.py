import numpy as np
import sys
from six import StringIO

from gym import spaces, utils
from gym.envs.toy_text import discrete

MAP = [
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

        n = len(map)
        m = len(map[0])
        sx = -1
        sy = -1
        for i in range(n):
            for j in range(m):
                if map[i][j] == 's':
                    sx = i
                    sy = j
                    map[i][j] = '0'
        self.map = map
        self.sx = sx
        self.sy = sy

        nS = n * m 
        nA = len(dirs)        
        isd = np.zeros(nS)
        isd[self.encode(sx, sy)]

        for si in range(n):
            for sj in range(m):
                if map[si][sj] == '1':
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
                    if map[dx][dy] == '0':
                        newstate = self.encode(dx, dy)
                    else:
                        newstate = self.encode(si, sj)
                    P[state][a].append((1.0, newstate, reward, done))
        
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def encode(self, sx, sy, m):
        return sx * m + sy

    def decode(self, i, m):
        return [i / m, i % m]

    def makeMap(self, map, sx, sy):
        map = map.copy()
        map[sx][sy] = 's'
        return map

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        map = self.makeMap(self.map, self.sx, self.sy)
        outfile.write('\n'.join(map) + '\n')

        # No need to return anything for human
        if mode != 'human':
            return outfile


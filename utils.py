import numpy as np
import config

# right, down, up, left
dirs = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])
dir_symbols = ['>', 'v', '^', '<']

def inMap(x, y):
    return x >= 0 and x < config.Map.Height and y >= 0 and y < config.Map.Width

def setCellValue(mazemap, x, y, value):
    for i in range(len(mazemap[x][y])):
        mazemap[x][y][i] = 0
    mazemap[x][y][value] = 1

def getCellValue(mazemap, x, y):
    for i in range(len(mazemap[x][y])):
        if mazemap[x][y][i] == 1:
            return i

def getDistance(sx, sy, tx, ty):
    return abs(sx - tx) + abs(sy - ty)

def categoricalSample(prob_n, np_random):
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

def initMazeMap():
    mazemap = []
    for i in range(config.Map.Height):
        mazemap.append([])
        for j in range(config.Map.Width):
            mazemap[i].append(np.zeros(4))
            setCellValue(mazemap, i, j, config.Cell.Empty)
    return mazemap

def displayMap(mazemap):
    output = ''
    for i in range(len(mazemap)):
        for j in range(len(mazemap[i])):
            output += str(getCellValue(mazemap, i, j))
        output += '\n'
    print output
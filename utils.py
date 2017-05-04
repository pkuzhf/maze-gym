import numpy as np
import config
import os

# right, down, up, left
dirs = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])
dir_symbols = ['>', 'v', '^', '<']

def inMap(x, y):
    return x >= 0 and x < config.Map.Height and y >= 0 and y < config.Map.Width

def equalCellValue(mazemap, x, y, value):
    return np.array_equal(mazemap[x][y], value)

def nequalCellValue(mazemap, x, y, value):
    return not np.array_equal(mazemap[x][y], value)

def setCellValue(mazemap, x, y, value):
    mazemap[x][y] = value

def getCellValue(mazemap, x, y):
    return mazemap[x][y]

def getDistance(sx, sy, tx, ty):
    return abs(sx - tx) + abs(sy - ty)


def findSourceAndTarget(mazemap):
    for i in range(len(mazemap)):
        for j in range(len(mazemap[i])):
            if equalCellValue(mazemap, i, j, config.Cell.Source):
                sx = i
                sy = j
            if equalCellValue(mazemap, i, j, config.Cell.Target):
                tx = i
                ty = j
    return [sx, sy, tx, ty]

def initMazeMap():
    mazemap = np.zeros([config.Map.Height, config.Map.Width, config.Cell.CellSize], dtype=np.int64)
    for i in range(config.Map.Height):
        for j in range(config.Map.Width):
            setCellValue(mazemap, i, j, config.Cell.Empty)
    return mazemap

def displayMap(mazemap):
    output = ''
    for i in range(config.Map.Height):
        for j in range(config.Map.Width):
            cell = getCellValue(mazemap, i, j)
            for k in range(config.Cell.CellSize):
                if cell[k]:
                    output += str(k)
                    break
        output += '\n'
    print output


def remove(path):
    if os.path.exists(path):
        os.remove(path)

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def removedirs(path):
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)


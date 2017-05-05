import numpy as np
import config
import os

# right, down, up, left
dirs = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])
dir_symbols = ['>', 'v', '^', '<']

class Cell:
    Empty  = np.asarray([1,0,0,0])
    Wall   = np.asarray([0,1,0,0])
    Source = np.asarray([0,0,1,0])
    Target = np.asarray([0,0,0,1])
    CellSize = 4

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
    [sx, sy, tx, ty] = [0, 0, 0, 0]
    for i in range(len(mazemap)):
        for j in range(len(mazemap[i])):
            if equalCellValue(mazemap, i, j, Cell.Source):
                sx = i
                sy = j
            if equalCellValue(mazemap, i, j, Cell.Target):
                tx = i
                ty = j
    return [sx, sy, tx, ty]

def initMazeMap():
    mazemap = np.zeros([config.Map.Height, config.Map.Width, Cell.CellSize], dtype=np.int64)
    for i in range(config.Map.Height):
        for j in range(config.Map.Width):
            setCellValue(mazemap, i, j, Cell.Empty)
    return mazemap

def nonempty_count(mazemap):
    not_empty_count = 0
    for i in range(0, config.Map.Height):
        for j in range(0, config.Map.Width):
            if nequalCellValue(mazemap, i, j, Cell.Empty):
                not_empty_count += 1
    return not_empty_count

def displayMap(mazemap):
    output = ''
    for i in range(config.Map.Height):
        for j in range(config.Map.Width):
            cell = getCellValue(mazemap, i, j)
            for k in range(Cell.CellSize):
                if cell[k]:
                    output += str(k)
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


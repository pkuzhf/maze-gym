import numpy as np
import config
import os
from keras.layers import Layer
import tensorflow as tf

# right, down, up, left
dirs = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])
dir_symbols = ['>', 'v', '^', '<']
map_symbols = ['0', '1', '2', '3']
#map_symbols = ['.', '#', 'S', 'T']

class Cell:
    CellSize = 4

    Empty  = 0
    Wall   = 1
    Source = 2
    Target = 3

    EmptyV  = np.array([1, 0, 0, 0])
    WallV   = np.array([0, 1, 0, 0])
    SourceV = np.array([0, 0, 1, 0])
    TargetV = np.array([0, 0, 0, 1])

    Value = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])]

def inMap(x, y):
    return x >= 0 and x < config.Map.Height and y >= 0 and y < config.Map.Width

def findSourceAndTarget(mazemap):
    sx, sy, tx, ty = -1, -1, -1, -1
    for i in range(config.Map.Height):
        for j in range(config.Map.Width):
            if mazemap[i, j, Cell.Source] == 1:
                sx = i
                sy = j
            if mazemap[i, j, Cell.Target] == 1:
                tx = i
                ty = j
    return sx, sy, tx, ty

def initMazeMap():
    mazemap = np.zeros([config.Map.Height, config.Map.Width, Cell.CellSize], dtype=np.int64)
    for i in range(config.Map.Height):
        for j in range(config.Map.Width):
            mazemap[i, j, Cell.Empty] = 1
    mazemap[0, 0] = Cell.SourceV
    mazemap[config.Map.Height-1, config.Map.Width-1] = Cell.TargetV
    return mazemap

def displayMap(mazemap):
    output = ''
    for i in range(config.Map.Height):
        for j in range(config.Map.Width):
            cell = mazemap[i, j]
            for k in range(Cell.CellSize):
                if cell[k]:
                    output += map_symbols[k]
        output += '\n'
    print output,

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

def get_tau(reward_for_prob_one_of_ten):
    return reward_for_prob_one_of_ten / -np.log(0.1)

def get_session():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

#def equalCellValue(mazemap, x, y, index):
#    return mazemap[x, y, index] == 1

#def nequalCellValue(mazemap, x, y, index):
#    return not mazemap[x, y, index] == 1

#def setCellValue(mazemap, x, y, index):
#    mazemap[x, y] = Cell.Value[index]

#def getCellValue(mazemap, x, y):
#    return mazemap[x, y]

#def getDistance(sx, sy, tx, ty):
#    return abs(sx - tx) + abs(sy - ty)


class qlogger(object):

    def __init__(self):

        self.minq = 1e20
        self.maxq = -1e20
        self.cur_minq = 1e20
        self.cur_maxq = 1e20

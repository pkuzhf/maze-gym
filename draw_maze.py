import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import *
import config
import sys
import numpy as np
import time

def draw_maze(mazemap, filename):
    
    path = config.Path.Figs
    makedirs(path)

    height = len(mazemap)
    width = len(mazemap[0])

    fig = plt.figure()

    ax = fig.add_subplot(111, aspect='equal')
    ax.set_axis_off()

    for i in range(height):
        for j in range(width):
            color = None
            hatch = None
            linewidth = 0
            linestyle = None
            if mazemap[i][j] == '0':
                color = 'white'
                linewidth = 0.1
                linestyle = 'dotted'
            elif mazemap[i][j] == '1':
                color = 'black'
            elif mazemap[i][j] == '2':
                hatch = '/'
            else:
                hatch = '/'
            p = patches.Rectangle((j / float(width), (height - i - 1) / float(height)), 1. / width, 1. / height,
                    facecolor = color,
                    linewidth = linewidth,
                    linestyle = linestyle,
                    hatch = hatch
                )
            ax.add_patch(p)

    p = patches.Rectangle((0, 0), 1, 1, fill = False)
    ax.add_patch(p)

    fig.savefig(path + '/' + filename + '.pdf', dpi=90, bbox_inches='tight')

mazemap = [[2, 0], [1, 3]]

def main():
    filepath = sys.argv[1]
    
    idx = 0
    if len(sys.argv) >= 3:
        idx = int(sys.argv[2]) - 1
    
    filename = 'default'
    if len(sys.argv) >= 4:
        filename = sys.argv[3]
    filename += '_' + time.strftime("%Y%m%d_%H%M%S")
    
    lines = open(filepath, 'r').readlines()
    mazemap = []
    n = len(lines[idx]) - 1
    for i in range(n):
        mazemap.append([])
        for j in range(n):
            c = lines[idx + i][j]
            if c == '.':
                c = '0'
            elif c == '#':
                c = '1'
            elif c == 'S':
                c = '2'
            elif c == 'T':
                c = '3'
            mazemap[i].append(c)
    print mazemap
    draw_maze(mazemap, filename)

if __name__ == "__main__":
    main()
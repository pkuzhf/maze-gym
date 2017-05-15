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
            alpha = None
            if mazemap[i][j] == '0':
                color = 'yellow'
                alpha = 0.5
                linewidth = 0.5
                linestyle = 'dotted'
            elif mazemap[i][j] == '1':
                color = 'black'
            elif mazemap[i][j] == '2':
                hatch = '/'
                color = 'green'
            else:
                hatch = '/'
                color = 'red'
            p = patches.Rectangle((j / float(width), (height - i - 1) / float(height)), 1. / width, 1. / height,
                    facecolor = color,
                    linewidth = linewidth,
                    linestyle = linestyle,
                    hatch = hatch,
                    alpha = alpha
                )
            ax.add_patch(p)

    p = patches.Rectangle((0, 0), 1, 1, fill = False)
    ax.add_patch(p)

    fig.savefig(path + '/' + filename + '.pdf', dpi=90, bbox_inches='tight')

mazemap = [[2, 0], [1, 3]]

def read_maze(filepath, idx):
    lines = open(filepath, 'r').readlines()
    n = len(lines[idx]) - 1
    mazemap = []
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
    return mazemap

def main():
    # args: filepath, line number (start from 1), outputfilename
    if len(sys.argv) == 1:
        args = [
            ['./logs/20170512_190434.dfs_path_8x8.log', 687097, 'dfs_8x8_500_68'], 
            ['./logs/20170512_190434.dfs_path_8x8.log', 687097, 'dfs_8x8_1500_68'],             
            ['./logs/20170512_190434.dfs_path_8x8.log', 490617, 'dfs_8x8_2500_58'], 
            ['./logs/20170512_190434.dfs_path_8x8.log', 687097, 'dfs_8x8_3500_68'], 
            ['./logs/20170512_190434.dfs_path_8x8.log', 887786, 'dfs_8x8_4500_86']
        ]
        for [filepath, idx, outputname] in args:
            mazemap = read_maze(filepath, idx - 1)
            draw_maze(mazemap, outputname)
        return

    filepath = sys.argv[1]
    
    idx = 0
    if len(sys.argv) >= 3:
        idx = int(sys.argv[2]) - 1
    
    mazemap = read_maze(filepath, idx)

    if len(sys.argv) >= 4:
        filename = sys.argv[3]
    else:
        filename += 'default_' + time.strftime("%Y%m%d_%H%M%S")
    
    draw_maze(mazemap, filename)

if __name__ == "__main__":
    main()
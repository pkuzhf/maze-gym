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
            #['./logs/20170512_190434.dfs_path_8x8.log', 87, 'dfs_8x8_1_16'],
            #['./logs/20170512_190434.dfs_path_8x8.log', 9913, 'dfs_8x8_50_16'],
            #['./logs/20170512_190434.dfs_path_8x8.log', 28891, 'dfs_8x8_150_14'],
            #['./logs/20170512_190434.dfs_path_8x8.log', 48298, 'dfs_8x8_250_22'],
            #['./logs/20170512_190434.dfs_path_8x8.log', 97598, 'dfs_8x8_500_58'],
            #['./logs/20170512_190434.dfs_path_8x8.log', 294164, 'dfs_8x8_1500_64'],
            #['./logs/20170512_190434.dfs_path_8x8.log', 687097, 'dfs_8x8_3500_68'],
            #['./logs/20170512_190434.dfs_path_8x8.log', 887786, 'dfs_8x8_5000_86'],

            #['./logs/20170512_185357.shortest_path_8x8.log', 87, 'shortest_8x8_1_14'],
            #['./logs/20170512_185357.shortest_path_8x8.log', 97502, 'shortest_8x8_500_14'],
            #['./logs/20170512_185357.shortest_path_8x8.log', 195911, 'shortest_8x8_1000_30'],
            #['./logs/20170512_185357.shortest_path_8x8.log', 392733, 'shortest_8x8_2000_32'],
            #['./logs/20170512_185357.shortest_path_8x8.log', 589787, 'shortest_8x8_3000_32'],
            #['./logs/20170512_185357.shortest_path_8x8.log', 785469, 'shortest_8x8_4000_36'],
            #['./logs/20170512_185357.shortest_path_8x8.log', 981931, 'shortest_8x8_5000_36'],
            #['./logs/20170512_185357.shortest_path_8x8.log', 1178465, 'shortest_8x8_6000_38'],

            #['./logs/env_dqn_8x8.20170513_115847.log', 443, 'dqn_8x8_1_14'],
            #['./logs/env_dqn_8x8.20170513_115847.log', 11819, 'dqn_8x8_500_14'],
            #['./logs/env_dqn_8x8.20170513_115847.log', 21917, 'dqn_8x8_1000_19'],
            #['./logs/env_dqn_8x8.20170513_115847.log', 31629, 'dqn_8x8_1500_200'],
            #['./logs/env_dqn_8x8.20170513_115847.log', 42037, 'dqn_8x8_2000_59'],
            #['./logs/env_dqn_8x8.20170513_115847.log', 82366, 'dqn_8x8_4000_30'],
            #['./logs/env_dqn_8x8.20170513_115847.log', 122473, 'dqn_8x8_6000_69'],
            #['./logs/env_dqn_8x8.20170513_115847.log', 162730, 'dqn_8x8_8000_87'],
            #['./logs/env_dqn_8x8.20170513_115847.log', 202867, 'dqn_8x8_10000_87'],

            # ['./logs/right_hand_path_8x8.20170513_112822.log', 50, 'rh_8x8_1_14'],
            # ['./logs/right_hand_path_8x8.20170513_112822.log', 100200, 'rh_8x8_5000_18'],
            # ['./logs/right_hand_path_8x8.20170513_112822.log', 200632, 'rh_8x8_10000_26'],
            # ['./logs/right_hand_path_8x8.20170513_112822.log', 301103, 'rh_8x8_15000_36'],
            # ['./logs/right_hand_path_8x8.20170513_112822.log', 401515, 'rh_8x8_20000_38'],
            # ['./logs/right_hand_path_8x8.20170513_112822.log', 602361, 'rh_8x8_30000_48'],
            # ['./logs/right_hand_path_8x8.20170513_112822.log', 803204, 'rh_8x8_40000_58'],
            # ['./logs/right_hand_path_8x8.20170513_112822.log', 1004133, 'rh_8x8_50000_70'],
            # ['./logs/right_hand_path_8x8.20170513_112822.log', 1204870, 'rh_8x8_60000_74'],

            #['./logs/20170512_183854.shortest_path_5x5.log', 414466, 'shortest_5x5_16'],
            #['./logs/20170513_002611.shortest_path_6x6.log', 88415, 'shortest_6x6_22'],
            #['./logs/20170513_002639.shortest_path_7x7.log', 165158, 'shortest_7x7_30'],
            #['./logs/20170512_185357.shortest_path_8x8.log', 909973, 'shortest_8x8_38'],

            #['./logs/20170513_003442.dfs_path_5x5.log', 2925402, 'dfs_5x5_28'],
            #['./logs/20170513_003600.dfs_path_6x6.log', 75145, 'dfs_6x6_44'],
            #['./logs/20170513_003613.dfs_path_7x7.log', 651524, 'dfs_7x7_64'],
            #['./logs/20170512_190434.dfs_path_8x8.log', 887786, 'dfs_8x8_86'],

            ['./logs/20170512_173414.right_hand_path_5x5.log', 1363450, 'rh_5x5_28'],
            ['./logs/20170512_173631.right_hand_path_6x6.log', 1586811, 'rh_6x6_40'],
            ['./logs/20170513_112411.right_hand_path_7x7.log', 1797476, 'rh_7x7_56'],
            ['./logs/right_hand_path_8x8.20170513_112822.log', 1204870, 'rh_8x8_74'],

            ['./logs/20170512_173414.right_hand_path_5x5.log', 1127, 'dqn_5x5_10'], #69289 #69250 #
            ['./logs/20170512_173414.right_hand_path_5x5.log', 69250, 'dqn_5x5_33_back'],
            ['./logs/20170512_173631.right_hand_path_6x6.log', 675677, 'dqn_6x6_16'],
            ['./logs/20170513_112411.right_hand_path_7x7.log', 528339, 'dqn_7x7_30'],
            ['./logs/20170513_112411.right_hand_path_7x7.log', 528305, 'dqn_7x7_30_back'],
            ['./logs/env_dqn_8x8.20170513_115847.log', 483102, 'dqn_8x8_87'],
            ['./logs/env_dqn_8x8.20170513_115847.log', 202867, 'dqn_8x8_87_back'],
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
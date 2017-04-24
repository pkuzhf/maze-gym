import numpy as np
import gym
import datetime

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

evaluate_log_file = '/home/zhf/drive200g/openaigym/code/evaluate.txt'

ENV_NAME = 'Maze-v0'
dirs = [[0, 1], [1, 0], [-1, 0], [0, -1]]

dir_symbols = ['>', 'v', '^', '<']

def setMapValue(mazemap, x, y, value):
    for i in range(len(mazemap[x][y])):
        mazemap[x][y][i] = 0
    mazemap[x][y][value] = 1

def getMapValue(mazemap, x, y):
    for i in range(len(mazemap[x][y])):
        if mazemap[x][y][i] == 1:
            return i

def getDistance(sx, sy, tx, ty):
    return abs(sx - tx) + abs(sy - ty)

def evaluate(model):
    n = 8
    m = 8
    mazemap = []
    wall_prob = 0
    for i in range(n):
        mazemap.append([])
        for j in range(m):
            mazemap[i].append(np.zeros(4))
            setMapValue(mazemap, i, j, np.random.binomial(1, wall_prob))
    value_memory = {}
    open(evaluate_log_file, 'w')
    for distance in range(1, 16):
        sum_score = 0
        for sx in range(n):
            for sy in range(m):
                if getMapValue(mazemap, sx, sy) == 1:
                    continue
                setMapValue(mazemap, sx, sy, 2)
                score = 0
                count = 0
                output = ''
                for tx in range(n):
                    for ty in range(m):
                        if getMapValue(mazemap, tx, ty) == 0 and getDistance(sx, sy, tx, ty) <= distance:
                            count += 1
                            setMapValue(mazemap, tx, ty, 3)
                            memory_id = str(sx) + '_' + str(sy) + '_' + str(tx) + '_' + str(ty)
                            if memory_id in value_memory:
                                dir_id = value_memory[memory_id]
                            else:
                                dir_id = np.array(model.predict(np.array([[mazemap]]))).argmax()
                                value_memory[memory_id] = dir_id
                            output += dir_symbols[dir_id]
                            setMapValue(mazemap, tx, ty, 0)
                            if getDistance(sx, sy, tx, ty) > getDistance(sx + dirs[dir_id][0], sy + dirs[dir_id][1], tx, ty):
                                score += 1
                sum_score += float(score) / count
                setMapValue(mazemap, sx, sy, 0)
        sum_score /= n * m
        print [distance, sum_score]
        f = open(evaluate_log_file, 'a')
        f.write(str(distance) + '\t' + str(sum_score) + '\n')
        f.close()

# Get the environment and extract the number of actions.
open(evaluate_log_file, 'w')
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
mazemap = env.reset()
nb_actions = env.action_space.n

n = len(mazemap)
m = len(mazemap[0])
# Next, we build a very simple model.
model = Sequential()
model.add(Reshape((8,8,4), input_shape=(1,8,8,4)))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

nround = 2000

for round in range(nround):
    print 'round ' + str(round) + '/' + str(nround)
    dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)
    evaluate(model)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights_{}.h5f'.format(ENV_NAME, datetime.datetime.now().isoformat()), overwrite=True)   

# Finally, evaluateObs our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)


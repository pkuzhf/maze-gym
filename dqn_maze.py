import numpy as np
import gym
import datetime
from gym.envs.toy_text.maze import AdversarialMazeEnv
from gym.envs.toy_text.maze_generator_env import MazeGeneratorEnv

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

import os
import config, utils, generator

MAZE_ENV_NAME = 'AdversarialMaze-v0'
GENERATOR_ENV_NAME = 'MazeGenerator-v0'

def evaluate(agent_model):
    n = config.Map.Height
    m = config.Map.Width
    mazemap = []
    for i in range(n):
        mazemap.append([])
        for j in range(m):
            mazemap[i].append(np.zeros(4))
            utils.setCellValue(mazemap, i, j, np.random.binomial(1, config.Map.WallDense))
    cell_value_memory = {}
    open(config.StrongMazeEnv.EvaluateFile, 'w')
    for distance in range(1, n + m):
        sum_score = 0
        for sx in range(n):
            for sy in range(m):
                if utils.getCellValue(mazemap, sx, sy) == config.Cell.Wall:
                    continue
                utils.setCellValue(mazemap, sx, sy, config.Cell.Source)
                score = 0
                count = 0
                output = ''
                for tx in range(n):
                    for ty in range(m):
                        if utils.getCellValue(mazemap, tx, ty) == config.Cell.Empty and utils.getDistance(sx, sy, tx, ty) <= distance:
                            count += 1
                            utils.setCellValue(mazemap, tx, ty, config.Cell.Target)
                            memory_id = str(sx) + '_' + str(sy) + '_' + str(tx) + '_' + str(ty)
                            if memory_id in cell_value_memory:
                                dir_id = cell_value_memory[memory_id]
                            else:
                                dir_id = np.array(agent_model.predict(np.array([[mazemap]]))).argmax()
                                cell_value_memory[memory_id] = dir_id
                            output += utils.dir_symbols[dir_id]
                            utils.setCellValue(mazemap, tx, ty, config.Cell.Empty)
                            if utils.getDistance(sx, sy, tx, ty) > utils.getDistance(sx + utils.dirs[dir_id][0], sy + utils.dirs[dir_id][1], tx, ty):
                                score += 1
                sum_score += float(score) / count
                utils.setCellValue(mazemap, sx, sy, config.Cell.Empty)
        sum_score /= n * m
        print [distance, sum_score]
        f = open(config.StrongMazeEnv.EvaluateFile, 'a')
        f.write(str(distance) + '\t' + str(sum_score) + '\n')
        f.close()

n = config.Map.Height
m = config.Map.Width
# Next, we build an agent model.
agent_model = Sequential()
agent_model.add(Reshape((8,8,4), input_shape=(1,8,8,4)))
agent_model.add(Conv2D(8, (3, 3), activation='relu'))
agent_model.add(Conv2D(8, (3, 3), activation='relu'))
agent_model.add(Conv2D(8, (3, 3), activation='relu'))
agent_model.add(MaxPooling2D(pool_size=(2, 2)))
# agent_model.add(Dropout(0.25))

agent_model.add(Flatten())
agent_model.add(Dense(256, activation='relu'))
# agent_model.add(Dropout(0.5))
agent_model.add(Dense(4, activation='softmax'))

print(agent_model.summary())

# Next, we build a generator model.
generator_model = Sequential()
generator_model.add(Reshape((8,8,4), input_shape=(1,8,8,4)))
generator_model.add(Conv2D(8, (3, 3), activation='relu'))
generator_model.add(Conv2D(8, (3, 3), activation='relu'))
generator_model.add(Conv2D(8, (3, 3), activation='relu'))
generator_model.add(MaxPooling2D(pool_size=(2, 2)))
# generator_model.add(Dropout(0.25))

generator_model.add(Flatten())
generator_model.add(Dense(256, activation='relu'))
# generator_model.add(Dropout(0.5))
generator_model.add(Dense(n * m + 1, activation='softmax'))

print(generator_model.summary())

map_generator = generator.GeneratorCNN(generator_model)

# Get the environment and extract the number of actions.
open(config.StrongMazeEnv.EvaluateFile, 'w')
np.random.seed(123)

#maze_env = gym.make(MAZE_ENV_NAME, map_generator)
maze_env = AdversarialMazeEnv(map_generator)
maze_env.seed(123)

#generator_env = gym.make(GENERATOR_ENV_NAME, agent_model, map_generator)
generator_env = MazeGeneratorEnv(agent_model, map_generator)
generator_env.seed(123)

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
maze_memory = SequentialMemory(limit=50000, window_length=1)
maze_policy = BoltzmannQPolicy()
maze_dqn = DQNAgent(model=agent_model, nb_actions=maze_env.action_space.n, memory=maze_memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=maze_policy)
maze_dqn.compile(Adam(lr=1e-3), metrics=['mae'])


generator_memory = SequentialMemory(limit=50000, window_length=1)
generator_policy = BoltzmannQPolicy()
generator_dqn = DQNAgent(model=generator_model, nb_actions=generator_env.action_space.n, memory=generator_memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=generator_policy)
generator_dqn.compile(Adam(lr=1e-3), metrics=['mae'])


# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

nround = 2000
timestamp = datetime.datetime.now().isoformat()
os.mkdir(timestamp)
for round in range(nround):
    print 'round ' + str(round) + '/' + str(nround)
    maze_dqn.fit(maze_env, nb_steps=500, visualize=True, verbose=2)
    evaluate(agent_model)
    maze_dqn.save_weights(timestamp + '/agent_model_weights_{}.h5f'.format(str(round)), overwrite=True)
    generator_dqn.fit(generator_env, nb_steps=500, visualize=True, verbose=2)
    generator_dqn.save_weights(timestamp + '/generator_model_weights_{}.h5f'.format(str(round)), overwrite=True)

# After training is done, we save the final weights.


# Finally, evaluateObs our algorithm for 5 episodes.
#dqn.test(env, nb_episodes=5, visualize=True)


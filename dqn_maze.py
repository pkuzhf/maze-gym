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

def evaluate(observation, model):
	output = ''
	score = 0
	n = len(observation[0][0])
	m = len(observation[0][0][0])
	tx = -1
	ty = -1
	for x in range(n):
		for y in range(m):
			if getMapValue(observation[0][0], x, y) == 2:
				setMapValue(observation[0][0], x, y, 0)
			if getMapValue(observation[0][0], x, y) == 3:
				tx = x
				ty = y
	for x in range(n):
		for y in range(m):
			if getMapValue(observation[0][0], x, y) == 0:
				setMapValue(observation[0][0], x, y, 2)
				dir_id = np.array(model.predict(observation)).argmax()
				output += dir_symbols[dir_id]
				setMapValue(observation[0][0], x, y, 0)
				if abs(x - tx) + abs(y - ty) > abs(x + dirs[dir_id][0] - tx) + abs(y + dirs[dir_id][1] - ty):
					score += 1
			else:
				output += str(getMapValue(observation[0][0], x, y))
			output += ' '
		output +=  '\n'
	return [output, score]

# Get the environment and extract the number of actions.
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

nround = 10000

for _ in range(nround):
	print 'round ' + str(_) + '/' + str(nround)
	dqn.fit(env, nb_steps=500, visualize=True, verbose=2)
	env = gym.make(ENV_NAME)
	env.seed(123)
	observation = np.array([[env.reset()]])
	[action_map, score] = evaluate(observation, model) 
	print action_map
	print score

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights_{}.h5f'.format(ENV_NAME, datetime.datetime.now().isoformat()), overwrite=True)   

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)


import numpy as np
import utils, config

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from gym.utils import seeding


def get_env_net():
    n = config.Map.Height
    m = config.Map.Width

    env_model = Sequential()
    env_model.add(Reshape((8, 8, 4), input_shape=(1, 8, 8, 4)))
    env_model.add(Conv2D(8, (3, 3), activation='relu'))
    env_model.add(Conv2D(8, (3, 3), activation='relu'))
    env_model.add(Conv2D(8, (3, 3), activation='relu'))
    env_model.add(MaxPooling2D(pool_size=(2, 2)))
    # generator_model.add(Dropout(0.25))
    env_model.add(Flatten())
    env_model.add(Dense(256, activation='relu'))
    # generator_model.add(Dropout(0.5))
    env_model.add(Dense(n * m, activation=None))

    print(env_model.summary())
    return env_model


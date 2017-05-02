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
    env_model.add(Dense(n * m, activation='softmax'))

    print(env_model.summary())
    return env_model


class env_generator:

    def __init__(self, model):
        self.model = model
        self.np_random, seed = seeding.np_random()

    def get_env_map(self, mazemap=None):

        if mazemap == None:
            mazemap = utils.initMazeMap()

        not_empty_count = 0
        for i in range(config.Map.Height):
            for j in range(config.Map.Width):
                if utils.getCellValue(mazemap, i, j) != config.Cell.Empty:
                    not_empty_count += 1

        while True:
            prob_n = self.model.predict(np.array([[mazemap]]))
            action = utils.categoricalSample(prob_n, self.np_random)
            if np.random.rand() < config.Generator.ExploreRate:
                action = np.random.randint(config.Map.Height * config.Map.Width)

            [x, y] = [action / config.Map.Width, action % config.Map.Width]
            if utils.getCellValue(mazemap, x, y) == config.Cell.Empty:
                if not_empty_count == 0:
                    utils.setCellValue(mazemap, x, y, config.Cell.Source)
                elif not_empty_count == 1:
                    utils.setCellValue(mazemap, x, y, config.Cell.Target)
                else:
                    utils.setCellValue(mazemap, x, y, config.Cell.Wall)
                not_empty_count += 1
            elif not_empty_count >= 2:
                break

        return mazemap
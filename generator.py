from keras.models import Sequential
import utils, config
from gym.utils import seeding
import numpy as np

class Generator(object):

    def __init__(self, model):
        self.model = model
        self.np_random, seed = seeding.np_random()

    def generate(self, mazemap = None):
        if mazemap == None:
            mazemap = utils.initMazeMap()
        return mazemap

class GeneratorCNN(Generator):

    def generate(self, mazemap = None):
        mazemap = super(GeneratorCNN, self).generate(mazemap)
        gamestep = 0
        for i in range(config.Map.Height):
            for j in range(config.Map.Width):
                if utils.getCellValue(mazemap, i, j) != config.Cell.Empty:
                    gamestep += 1
        while True:
            prob_n = self.model.predict(np.array([[mazemap]]))
            action = utils.categoricalSample(prob_n, self.np_random)
            if np.random.rand() < config.Generator.ExploreRate:
                action = np.random.randint(config.Map.Height * config.Map.Width + 1)
            if action == config.Map.Height * config.Map.Width or gamestep == config.Map.Height * config.Map.Width:
                if gamestep >= 2:
                    done = True
                else:
                    done = False
            else:
                done = False
                [x, y] = [action / config.Map.Width, action % config.Map.Width]
                if utils.getCellValue(mazemap, x, y) == config.Cell.Empty:
                    if gamestep == 0:
                        utils.setCellValue(mazemap, x, y, config.Cell.Source)
                    elif gamestep == 1:
                        utils.setCellValue(mazemap, x, y, config.Cell.Target)
                    else:
                        utils.setCellValue(mazemap, x, y, config.Cell.Wall)
                    gamestep += 1
            if done:
                break
        return mazemap
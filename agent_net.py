import config
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D


def get_agent_net():

    agent_model = Sequential()
    agent_model.add(Reshape((8,8,4), input_shape=(1,8,8,4)))
    agent_model.add(Conv2D(8, (3, 3), activation='relu'))
    agent_model.add(Conv2D(8, (3, 3), activation='relu'))
    agent_model.add(Conv2D(8, (3, 3), activation='relu'))
    agent_model.add(MaxPooling2D(pool_size=(2, 2)))
    agent_model.add(Flatten())
    agent_model.add(Dense(256, activation='relu'))
    agent_model.add(Dense(4, activation=None))

    print(agent_model.summary())
    return agent_model

    n = config.Map.Height
    m = config.Map.Width
    d = config.Cell.CellSize

    curdim = 8

    agent_model = Sequential()
    agent_model.add(Reshape((m, n, d), input_shape=(1, m, n, d)))

    for i in range(5):
        agent_model.add(Conv2D(curdim, (3, 3), activation='relu', padding='same'))
        curdim = min(64, curdim*2)

    agent_model.add(MaxPooling2D(pool_size=(m, n)))
    agent_model.add(Flatten())
    agent_model.add(Dense(4, activation=None))

    print(agent_model.summary())
    return agent_model
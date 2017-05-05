import config, utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D


def get_agent_net():

    return get_agent_net0()

    n = config.Map.Height
    m = config.Map.Width
    d = utils.Cell.CellSize

    curdim = 32
    use_bn = False

    agent_model = Sequential()
    agent_model.add(Reshape((m, n, d), input_shape=(1, m, n, d)))

    for i in range(5):
        agent_model.add(Conv2D(filters=curdim, kernel_size=(3, 3), padding='same'))
        if use_bn: agent_model.add(BatchNormalization())
        agent_model.add(Activation(activation='relu'))
        curdim = min(32, curdim*2)

    agent_model.add(GlobalAveragePooling2D())

    for i in range(3):
        agent_model.add(Dense(100))
        if use_bn: agent_model.add(BatchNormalization())
        agent_model.add(Activation(activation='relu'))

    agent_model.add(Dense(4, activation=None))

    print 'agent model:'
    print(agent_model.summary())
    return agent_model

def get_agent_net0():

    n = config.Map.Height
    m = config.Map.Width
    d = utils.Cell.CellSize

    agent_model = Sequential()
    agent_model.add(Reshape((m, n, d), input_shape=(1, m, n, d)))

    agent_model.add(Conv2D(16, (3, 3), activation='relu'))
    agent_model.add(Conv2D(32, (3, 3), activation='relu'))
    agent_model.add(Conv2D(64, (3, 3), activation='relu'))
    agent_model.add(Flatten())
    agent_model.add(Dense(200, activation='relu'))
    agent_model.add(Dense(150, activation='relu'))
    agent_model.add(Dense(100, activation='relu'))
    agent_model.add(Dense(50, activation='relu'))
    agent_model.add(Dense(4, activation=None))

    print 'agent model:'
    print(agent_model.summary())
    return agent_model
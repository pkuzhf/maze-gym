import config, utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D


def get_agent_net():

    #return get_agent_net0()

    n = config.Map.Height
    m = config.Map.Width
    d = utils.Cell.CellSize

    curdim = 32
    use_bn = False

    agent_model = Sequential()
    agent_model.add(Reshape((m, n, d), input_shape=(1, m, n, d)))

    agent_model.add(Conv2D(filters=curdim, kernel_size=(5, 5), padding='same', activation='relu'))

    for i in range(11):

        if use_bn:
            agent_model.add(Conv2D(filters=curdim, kernel_size=(3, 3), padding='same'))
            agent_model.add(BatchNormalization())
            agent_model.add(Activation(activation='relu'))
        else:
            agent_model.add(Conv2D(filters=curdim, kernel_size=(3, 3), padding='same', activation='relu'))

        curdim = min(64, curdim*1)

    agent_model.add(Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu'))
    agent_model.add(Flatten())

    curfilter = 256
    for i in range(0):

        if use_bn:
            agent_model.add(Dense(curfilter))
            agent_model.add(BatchNormalization())
            agent_model.add(Activation(activation='relu'))
        else:
            agent_model.add(Dense(curfilter, activation='relu'))

        curfilter = max(4 * 2, curfilter // 2)

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
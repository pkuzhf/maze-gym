import config, utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D


def get_env_net():

    #return get_env_net0()

    n = config.Map.Height
    m = config.Map.Width
    d = utils.Cell.CellSize

    curdim = 32
    use_bn = False

    env_model = Sequential()
    env_model.add(Reshape((m, n, d), input_shape=(1, m, n, d)))

    env_model.add(Conv2D(filters=curdim, kernel_size=(5, 5), padding='same', activation='relu'))

    for i in range(11):

        if use_bn:
            env_model.add(Conv2D(filters=curdim, kernel_size=(3, 3), padding='same'))
            env_model.add(BatchNormalization())
            env_model.add(Activation(activation='relu'))
        else:
            env_model.add(Conv2D(filters=curdim, kernel_size=(3, 3), padding='same', activation='relu'))

        curdim = min(64, curdim * 1)

    env_model.add(Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu'))
    env_model.add(Flatten())

    curfilter = 256
    for i in range(0):

        if use_bn:
            env_model.add(Dense(curfilter))
            env_model.add(BatchNormalization())
            env_model.add(Activation(activation='relu'))
        else:
            env_model.add(Dense(curfilter, activation='relu'))

        curfilter = max((n*m+1)*2, curfilter // 2)

    env_model.add(Dense(n * m + 1, activation=None))

    print 'env model:'
    print(env_model.summary())
    return env_model

def get_env_net0():

    m = config.Map.Height
    n = config.Map.Width
    d = utils.Cell.CellSize

    env_model = Sequential()
    env_model.add(Reshape((m, n, d), input_shape=(1, m, n, d)))

    env_model.add(Conv2D(16, (3, 3), activation='relu'))
    env_model.add(Conv2D(32, (3, 3), activation='relu'))
    env_model.add(Conv2D(64, (3, 3), activation='relu'))
    env_model.add(Flatten())
    env_model.add(Dense(200, activation='relu'))
    env_model.add(Dense(150, activation='relu'))
    env_model.add(Dense(100, activation='relu'))
    env_model.add(Dense(m * n + 1, activation=None))

    print 'env model:'
    print(env_model.summary())
    return env_model


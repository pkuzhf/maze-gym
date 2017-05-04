import config
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D


def get_env_net():

    return get_env_net0()

    n = config.Map.Height
    m = config.Map.Width
    d = config.Cell.CellSize

    curdim = 8

    env_model = Sequential()
    env_model.add(Reshape((m, n, d), input_shape=(1, m, n, d)))

    for i in range(5):
        env_model.add(Conv2D(filters=curdim, kernel_size=(3, 3), padding='same'))
        env_model.add(BatchNormalization())
        env_model.add(Activation(activation='relu'))
        curdim = min(64, curdim * 2)

    env_model.add(GlobalAveragePooling2D())

    for i in range(3):
        env_model.add(Dense(100))
        env_model.add(BatchNormalization())
        env_model.add(Activation(activation='relu'))

    env_model.add(Dense(n * m, activation=None))

    print 'env model:'
    print(env_model.summary())
    return env_model

def get_env_net0():

    n = config.Map.Height
    m = config.Map.Width

    env_model = Sequential()
    env_model.add(Reshape((8, 8, 4), input_shape=(1, 8, 8, 4)))
    env_model.add(Conv2D(8, (3, 3), activation='relu'))
    env_model.add(Conv2D(8, (3, 3), activation='relu'))
    env_model.add(Conv2D(8, (3, 3), activation='relu'))
    env_model.add(Flatten())
    env_model.add(Dense(256, activation='relu'))
    env_model.add(Dense(n * m, activation='softmax'))

    print 'env model:'
    print(env_model.summary())
    return env_model

import config
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D


def get_env_net():

    n = config.Map.Height
    m = config.Map.Width
    d = config.Cell.CellSize

    curdim = 8

    env_model = Sequential()
    env_model.add(Reshape((m, n, d), input_shape=(1, m, n, d)))

    for i in range(5):
        env_model.add(Conv2D(curdim, (3, 3), activation='relu', padding='same'))
        curdim = min(64, curdim * 2)

    env_model.add(MaxPooling2D(pool_size=(m, n)))
    env_model.add(Flatten())
    env_model.add(Dense(n * m, activation=None))

    print(env_model.summary())
    return env_model


import config, utils
from keras import backend as K
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, LeakyReLU, PReLU, merge, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.activations import relu
from keras.initializers import *
from scaleshift import Scaleshift

def get_env_net():

    n = config.Map.Height
    m = config.Map.Width
    d = utils.Cell.CellSize

    use_bn = True

    observation = Input(shape=(1, m, n, d), name='observation_input')
    x = Reshape((m, n, d))(observation)

    list = [32, 32, 32, 32, 32]
    for curdim in list:
        x = Conv2D(filters=curdim, kernel_size=(3, 3), padding='same')(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)

        x = Conv2D(filters=curdim, kernel_size=(1, 1), padding='same')(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    #x = Dense(256, activation=None)(x)
    actions = Dense(m*n+1)(x)

    env_actor = Model(inputs=observation, outputs=actions)

    print('env model:')
    print(env_actor.summary())
    return env_actor
import config, utils
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation, LeakyReLU, PReLU
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.initializers import *

def get_agent_net():

    n = config.Map.Height
    m = config.Map.Width
    d = utils.Cell.CellSize

    use_bn = True

    observation = Input(shape=(1, m, n, d), name='observation_input')
    x = Reshape((m, n, d))(observation)

    list = [32, 32, 32]
    for curdim in list:
        x = Conv2D(filters=curdim, kernel_size=(3, 3), padding='same')(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)

        # x = Conv2D(filters=curdim, kernel_size=(1, 1), padding='same')(x)
        # if use_bn:
        #    x = BatchNormalization()(x)
        # x = Activation(activation='relu')(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)

    x = Dense(256)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.1)(x)

    actions = Dense(4)(x)

    agent_model = Model(inputs=observation, outputs=actions, name='agent')

    print('agent model:')
    print(agent_model.summary())
    return agent_model
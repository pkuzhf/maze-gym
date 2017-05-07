import config, utils
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D


def get_agent_net():

    #return get_agent_net0()

    n = config.Map.Height
    m = config.Map.Width
    d = utils.Cell.CellSize

    curdim = 32
    use_bn = False
    actfn = 'relu'

    observation = Input(shape=(1, m, n, d), name='observation_input')

    x = Reshape((m, n, d))(observation)

    for i in range(3):

        if use_bn:
            x = Conv2D(filters=curdim, kernel_size=(3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation(activation=actfn)(x)
        else:
            x = Conv2D(filters=curdim, kernel_size=(3, 3), padding='same', activation=actfn)(x)

        curdim = min(64, curdim * 1)

    #x = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(x)

    x = Flatten()(x)
    #x = Dense(256, activation=None)(x)

    actions = Dense(4, activation=None)(x)

    agent_model = Model(input=observation, output=actions)

    print 'agent model:'
    print(agent_model.summary())
    return agent_model

def get_agent_net0():

    n = config.Map.Height
    m = config.Map.Width
    d = utils.Cell.CellSize

    agent_model = Sequential()
    agent_model.add(Reshape((m, n, d), input_shape=(1, m, n, d)))

    agent_model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    agent_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    agent_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    agent_model.add(Flatten())
    agent_model.add(Dense(200, activation='relu'))
    agent_model.add(Dense(150, activation='relu'))
    agent_model.add(Dense(100, activation='relu'))
    agent_model.add(Dense(50, activation='relu'))
    agent_model.add(Dense(4, activation=None))

    print 'agent model:'
    print(agent_model.summary())
    return agent_model
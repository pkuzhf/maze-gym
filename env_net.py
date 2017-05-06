import config, utils
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation, merge
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


def get_env_actor():

    n = config.Map.Height
    m = config.Map.Width
    d = utils.Cell.CellSize

    curdim = 32
    use_bn = False

    env_actor = Sequential()
    env_actor.add(Reshape((m, n, d), input_shape=(1, m, n, d)))

    #env_actor.add(Conv2D(filters=curdim, kernel_size=(5, 5), padding='same', activation='relu'))

    for i in range(0):

        if use_bn:
            env_actor.add(Conv2D(filters=curdim, kernel_size=(3, 3), padding='same'))
            env_actor.add(BatchNormalization())
            env_actor.add(Activation(activation='relu'))
        else:
            env_actor.add(Conv2D(filters=curdim, kernel_size=(3, 3), padding='same', activation='relu'))

        curdim = min(64, curdim * 1)

    #env_actor.add(Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu'))
    env_actor.add(Flatten())

    curfilter = 32
    for i in range(1):

        if use_bn:
            env_actor.add(Dense(curfilter))
            env_actor.add(BatchNormalization())
            env_actor.add(Activation(activation='relu'))
        else:
            env_actor.add(Dense(curfilter, activation='relu'))

        curfilter = max((n*m+1)*2, curfilter // 2)

    env_actor.add(Dense(n * m + 1, activation=None))

    print 'env actor:'
    print(env_actor.summary())
    return env_actor


def get_env_critic():

    n = config.Map.Height
    m = config.Map.Width
    d = utils.Cell.CellSize

    action_input = Input(shape=(n * m + 1,), name='action_input')
    action_ = Dense(m * n, activation='relu')(action_input)
    action_ = Reshape((m, n, 1))(action_)

    observation = Input(shape=(1, m, n, d), name='observation_input')
    observation_ = Reshape((m, n, d))(observation)

    x = merge([action_, observation_], mode='concat', concat_axis=3)

    curdim = 32
    use_bn = False

    #x = Conv2D(filters=curdim, kernel_size=(5, 5), padding='same', activation='relu')(x)

    for i in range(0):

        if use_bn:
            x = Conv2D(filters=curdim, kernel_size=(3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation(activation='relu')(x)
        else:
            x = Conv2D(filters=curdim, kernel_size=(3, 3), padding='same', activation='relu')(x)

        curdim = min(64, curdim * 1)

    #x = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = Flatten()(x)

    curfilter = 32
    for i in range(1):

        if use_bn:
            x = Dense(curfilter)(x)
            x = BatchNormalization()(x)
            x = Activation(activation='relu')(x)
        else:
            x = Dense(curfilter, activation='relu')(x)

        curfilter = max(1 * 2, curfilter // 2)

    x = Dense(1, activation=None)(x)

    critic = Model(input=[action_input, observation], output=x)

    print 'env critic:'
    print(critic.summary())
    return critic, action_input


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


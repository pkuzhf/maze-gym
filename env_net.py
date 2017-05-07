import config, utils
from keras import backend as K
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation, merge
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D

def get_env_net():

    #return get_env_net0()
    return get_env_actor()

def get_env_actor():

    n = config.Map.Height
    m = config.Map.Width
    d = utils.Cell.CellSize

    curdim = 32
    use_bn = False
    actfn = 'relu'

    observation = Input(shape=(1, m, n, d), name='observation_input')
   
    x = Reshape((m, n, d))(observation)
    #x = Conv2D(filters=curdim, kernel_size=(5, 5), padding='same', activation=actfn)(x)

    for i in range(3):

        if use_bn:
            x=Conv2D(filters=curdim, kernel_size=(3, 3), padding='same')(x)
            x=BatchNormalization()(x)
            x=Activation(activation=actfn)(x)
        else:
            x=Conv2D(filters=curdim, kernel_size=(3, 3), padding='same', activation=actfn)(x)

        curdim = min(64, curdim * 1)

    end_action=Flatten()(x)
    end_action=Dense(1)(end_action)

    actions=Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation=actfn)(x)
    actions=Flatten()(actions)

    actions=merge([actions, end_action], mode='concat', concat_axis=1)

    env_actor = Model(input=observation, output=actions)

    print 'env actor:'
    print(env_actor.summary())
    return env_actor


def get_env_critic():

    n = config.Map.Height
    m = config.Map.Width
    d = utils.Cell.CellSize

    curdim = 32
    use_bn = False
    actfn = 'relu'

    action_input = Input(shape=(n * m + 1,), name='action_input')
    #a = K.zeros([m, n])
    #a[action_input] =

    action_ = Dense(m * n, activation=actfn)(action_input)
    action_ = Reshape((m, n, 1))(action_)

    observation = Input(shape=(1, m, n, d), name='observation_input')
    observation_ = Reshape((m, n, d))(observation)

    x = merge([action_, observation_], mode='concat', concat_axis=3)

    #x = Conv2D(filters=curdim, kernel_size=(5, 5), padding='same', activation=actfn)(x)

    for i in range(11):

        if use_bn:
            x = Conv2D(filters=curdim, kernel_size=(3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation(activation=actfn)(x)
        else:
            x = Conv2D(filters=curdim, kernel_size=(3, 3), padding='same', activation=actfn)(x)

        curdim = min(64, curdim * 1)

    x = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation=actfn)(x)
    x = Flatten()(x)

    x = Dense(1, activation=None)(x)

    critic = Model(input=[action_input, observation], output=x)

    print 'env critic:'
    print(critic.summary())
    return critic, action_input


def get_env_net0():

    m = config.Map.Height
    n = config.Map.Width
    d = utils.Cell.CellSize
    actfn = actfn

    env_model = Sequential()
    env_model.add(Reshape((m, n, d), input_shape=(1, m, n, d)))

    env_model.add(Conv2D(16, (3, 3), activation=actfn))
    env_model.add(Conv2D(32, (3, 3), activation=actfn))
    env_model.add(Conv2D(64, (3, 3), activation=actfn))
    env_model.add(Flatten())
    env_model.add(Dense(200, activation=actfn))
    env_model.add(Dense(150, activation=actfn))
    env_model.add(Dense(100, activation=actfn))
    env_model.add(Dense(m * n + 1, activation=None))

    print 'env model:'
    print(env_model.summary())
    return env_model


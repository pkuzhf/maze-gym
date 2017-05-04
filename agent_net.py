from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D


def get_agent_net():

    agent_model = Sequential()
    agent_model.add(Reshape((8, 8, 4), input_shape=(1,8,8,4)))
    agent_model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    agent_model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    agent_model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
    agent_model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
    agent_model.add(MaxPooling2D(pool_size=(2, 2)))
    agent_model.add(Flatten())
    agent_model.add(Dense(4, activation=None))

    print(agent_model.summary())
    return agent_model
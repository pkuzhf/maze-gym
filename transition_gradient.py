import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import copy
import config, utils
import numpy as np
import gym
from utils import *
from gym import spaces
from gym.utils import seeding
from collections import deque
from keras.layers import Dense, Lambda, Activation
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

from keras.layers import Input, Dense, Lambda, Layer, Reshape
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.merge import multiply, add
import sys
import config
import profile
from utils import *
import datetime
import numpy as np
from env_net import *
from env_gym import ENV_GYM
from agent_net import *
from agent_gym import ADVERSARIAL_AGENT_GYM
from keras.optimizers import *
from rl.agents.dqn import DQNAgent as DQN
from policy import *
from mydqn import myDQNAgent as mDQN
from rl.memory import SequentialMemory, RingBuffer




class TransitionGradientENV(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.action_space = spaces.Discrete(config.Map.Height * config.Map.Width + 1)

        t = ()
        for i in range(config.Map.Height * config.Map.Width):
            t += (spaces.Discrete(4),)
        self.observation_space = spaces.Tuple(t)

        self._seed()

        self.env = None
        self.agent = None
        self.mask = None
        self.gamestep = 0
        self.invalid_count = 0
        self.conflict_count = 0
        self.max_reward = -1e20
        self.reward_his = deque(maxlen=1000)
        self.global_step = 0

        self.used_agent = False
        self.agent_policy = [0.25] * 4
        self.agent_action_size = 4
        self.state_size = config.Map.Height * config.Map.Width * 4
        self.transition_size = config.Map.Height * config.Map.Width
        self.discount_factor = .99  # decay rate
        self.learning_rate = 0.1
        self.latent_dim = 16

        self.model = self.build_model()
        self.optimizer = self.optimizer()
        self.states, self.actions, self.rewards = [[],[],[]], [], []

    def build_model(self):
        self.noise = Input(shape=(self.latent_dim,))
        self.current_pos = Input(shape=(self.transition_size,))
        self.potential_pos = Input(shape=(self.transition_size,))
        self.last_prob = Input(shape=(1,))

        x = Dense(128, activation='relu', kernel_initializer='glorot_uniform')(self.noise)
        x = Dense(64, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dense(self.transition_size - 1, kernel_initializer='glorot_uniform')(x)
        self.tmp = x
        x = Activation('softmax')(x)
        x = Lambda(lambda xx: K.concatenate([xx, self.last_prob]))(x)
        self.probs = x
        x = Lambda(lambda xx: 1. - xx)(x)
        potential_prob = multiply([self.potential_pos, x])
        y = Lambda(lambda xx: 1. - K.tile(K.max(xx, 1), self.transition_size), output_shape=(self.transition_size,))(potential_prob)
        y = Reshape((self.transition_size,))(y)
        current_prob = multiply([self.current_pos, y])
        o = add([potential_prob, current_prob])
        model = Model(inputs=[self.noise, self.current_pos, self.potential_pos, self.last_prob], outputs=o)
        model.summary()

        return model

    def optimizer(self):
        action = K.placeholder(shape=[None, self.transition_size])
        discounted_rewards = K.placeholder(shape=[None, ])
        eps = 10e-7
        good_prob = K.clip(K.sum(action * self.model.output, axis=1), eps, 1.- eps)
        eligibility = K.log(good_prob) * discounted_rewards
        loss = K.sum(eligibility)

        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.noise, self.current_pos, self.potential_pos, self.last_prob, action, discounted_rewards], [loss, self.probs, self.tmp], updates=updates)
        return train

    def get_action(self, noise, current_pos_onehot, potential_pos_onehot):
        policy = self.model.predict([noise, current_pos_onehot, potential_pos_onehot, np.array([[0.]])], batch_size=1)
        policy = policy.flatten()
        return np.random.choice(self.transition_size, 1, p=policy)[0]

    def get_agent_action(self, state):
        state = copy.deepcopy(state)
        action = self.agent.forward(state)
        # print('action: ', action)
        return action

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def memory(self, state, action, reward):
        # [noise, current_pos_onehot, potential_pos_onehot]
        self.states[0].append(state[0][0])
        self.states[1].append(state[1][0])
        self.states[2].append(state[2][0])
        self.rewards.append(reward)
        act = np.zeros(self.transition_size)
        act[action] = 1
        self.actions.append(act)

    def train_episodes(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        last_probs = np.zeros((len(self.actions), 1))
        loss, map_probs, tmp = self.optimizer([self.states[0], self.states[1], self.states[2],last_probs, self.actions, discounted_rewards])
        # if self.global_step  % 100 == 0:
        print('loss: ', loss)
        print('map probs:')
        print(map_probs[0].reshape((3,3)))
            # print(tmp[0])
        print('===========================')
        self.states, self.actions, self.rewards = [[],[],[]], [], []

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)

    def _reset(self):
        self.gamestep = 0
        self.invalid_count = 0
        self.conflict_count = 0
        self.mazemap = utils.initMazeMap()
        [sx, sy, tx, ty] = utils.findSourceAndTarget(self.mazemap)
        self.source = np.array([sx, sy])
        self.target = np.array([tx, ty])
        return self.mazemap

    def _getmask(self, mazemap):
        mask = np.zeros(self.action_space.n)
        for i in range(config.Map.Height):
            for j in range(config.Map.Width):
                if not mazemap[i, j, utils.Cell.Empty]:
                    mask[i * config.Map.Width + j] = 1
        return mask

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def agent_step(self, action):
        done = False
        if self.gamestep >= config.Game.MaxGameStep:
            done = True

        reward = -1
        noise, current_pos_onehot, potential_pos_onehot = get_inputs_from_state_and_agent_action(self.mazemap, action, self.latent_dim, self.transition_size)
        new_pos = self.get_action(noise, current_pos_onehot, potential_pos_onehot)
        x, y = new_pos / config.Map.Width, new_pos % config.Map.Width
        new_source = [x, y]

        if self.mazemap[new_source[0], new_source[1], utils.Cell.Target]:
            done = True
        self.mazemap[self.source[0], self.source[1]] = utils.Cell.EmptyV
        self.mazemap[new_source[0], new_source[1]] = utils.Cell.SourceV
        self.source = new_source
        self.gamestep += 1
        return copy.deepcopy(self.mazemap), reward, done, new_pos, noise, current_pos_onehot, potential_pos_onehot

    def _step(self, action):
        done = False
        if self.gamestep >= config.Game.MaxGameStep:
            done = True

        reward = -1
        noise, current_pos_onehot, potential_pos_onehot = get_inputs_from_state_and_agent_action(self.mazemap, action,
                                                                                                 self.latent_dim,
                                                                                                 self.transition_size)
        new_pos = self.get_action(noise, current_pos_onehot, potential_pos_onehot)
        x, y = new_pos / config.Map.Width, new_pos % config.Map.Width
        new_source = [x, y]

        if self.mazemap[new_source[0], new_source[1], utils.Cell.Target]:
            done = True
        self.mazemap[self.source[0], self.source[1]] = utils.Cell.EmptyV
        self.mazemap[new_source[0], new_source[1]] = utils.Cell.SourceV
        self.source = new_source
        self.gamestep += 1
        return copy.deepcopy(self.mazemap), reward, done, {}


def get_pos_onehot(index):
    onehot_vector = np.zeros(config.Map.Height * config.Map.Width)
    onehot_vector[index] = 1.
    return onehot_vector

def get_potential_pos_onehot(index, a):
    x, y = index / config.Map.Width, index % config.Map.Width
    if a == 0:
        if y < (config.Map.Width - 1):
            y += 1
    if a == 1:
        if x < (config.Map.Height - 1):
            x += 1
    if a == 2:
        if x > 0:
            x -= 1
    if a == 3:
        if y > 0:
            y -= 1
    new_index = x * config.Map.Width + y
    return get_pos_onehot(new_index)


def get_inputs_from_state_and_agent_action(state, action, latent_dim, transition_size):
    [sx, sy, _, _] = utils.findSourceAndTarget(state)
    noise = np.ones(latent_dim).reshape((1, latent_dim))
    index = sx * config.Map.Width + sy
    current_pos_onehot = get_pos_onehot(index).reshape((1, transition_size))
    potential_pos_onehot = get_potential_pos_onehot(index, action).reshape((1, transition_size))
    return noise, current_pos_onehot, potential_pos_onehot


def train_env(env_gym):
    env_gym.reset()
    scores, episodes = [], []
    EPISODES = 200
    env_gym.agent.training = True
    for e in range(EPISODES):
        done = False
        score = 0
        state = env_gym.reset()
        while not done:
            # fresh env
            env_gym.global_step += 1

            # RL choose action based on observation and go one step
            action = env_gym.get_agent_action(state)
            next_state, reward, done, new_pos, noise, current_pos_onehot, potential_pos_onehot = env_gym.agent_step(
                action)
            # action_vector = np.zeros((env_gym.agent_action_size,))
            # action_vector[action] = 1
            env_gym.memory([noise, current_pos_onehot, potential_pos_onehot], new_pos, reward)
            score += reward
            state = next_state

            if done:
                env_gym.train_episodes()
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "  score:", score, "  time_step:", env_gym.global_step)

        if e % 100 == 0:
            pass
            env_gym.save_model("./models/transition_gradient.h5")

class CleanableMemory(SequentialMemory):
    def clear(self):
        self.actions = RingBuffer(self.limit)
        self.rewards = RingBuffer(self.limit)
        self.terminals = RingBuffer(self.limit)
        self.observations = RingBuffer(self.limit)


def main():
    argv = '\n\n'
    for arg in sys.argv:
        argv += arg + ' '
    print(argv)

    if len(sys.argv) >= 2:
        task_name = sys.argv[1]
    else:
        task_name = 'default'

    if 'dqn5' in task_name:
        config.Game.AgentAction = 5

    config.Game.Type = task_name

    if len(sys.argv) >= 4:
        config.Map.Height = int(sys.argv[2])
        config.Map.Width = int(sys.argv[3])
    config.Map.Height = 3
    config.Map.Width = 3
    # np.random.seed(config.Game.Seed)

    env_gym = TransitionGradientENV()

    agent_net = get_agent_net()
    agent_memory = CleanableMemory(limit=config.Training.AgentBufferSize, window_length=1)

    agent_policy = EpsABPolicy(policyA=GreedyQPolicy(), policyB=RandomPolicy(), eps_forB=config.Training.AgentTrainEps,
                               half_eps_step=config.Training.AgentTrainEps_HalfStep,
                               eps_min=config.Training.AgentTrainEps_Min)
    agent_test_policy = GreedyQPolicy()

    agent = mDQN(name='agent', model=agent_net, batch_size=config.Training.BatchSize, delta_clip=10, gamma=1.0,
                 nb_steps_warmup=config.Training.AgentWarmup,
                 target_model_update=config.Training.AgentTargetModelUpdate,
                 enable_dueling_network=True, policy=agent_policy, test_policy=agent_test_policy,
                 nb_actions=4, memory=agent_memory)

    agent.compile(Adam(lr=config.Training.AgentLearningRate))
    env_gym.agent = agent
    for _ in range(50):
        print('Traning Agent\n\n')
        agent.memory.clear()
        agent.fit(env_gym, nb_episodes=100, min_steps=100, visualize=False, verbose=2)
        print('Traning Env\n\n')
        train_env(env_gym)


if __name__ == "__main__":
    main()

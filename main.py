import config
from utils import *
import datetime
import numpy as np
from env_gym import ENV_GYM
from agent_gym import ADVERSARIAL_AGENT_GYM
from keras.optimizers import Adam
from rl.core import Processor
from rl.agents.dqn import DQNAgent as DQN
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from agent_net import get_agent_net
from env_net import get_env_net


n = config.Map.Height
m = config.Map.Width
np.random.seed(123)

env_net = get_env_net()
agent_net = get_agent_net()

env_gym = ENV_GYM(agent_net, env_net)
env_gym.seed(123)

agent_gym = ADVERSARIAL_AGENT_GYM(env_gym)
agent_gym.seed(123)

env_memory = SequentialMemory(limit=50000, window_length=1)
env_dqn = DQN(model=env_net, nb_actions=env_gym.action_space.n, memory=env_memory, nb_steps_warmup=10, target_model_update=1e-2,
              policy=BoltzmannQPolicy(), test_policy=BoltzmannQPolicy())
env_dqn.compile(Adam(lr=1e-3), metrics=['mae'])

agent_memory = SequentialMemory(limit=50000, window_length=1)
agent_dqn = DQN(model=agent_net, nb_actions=agent_gym.action_space.n, memory=agent_memory, nb_steps_warmup=10, target_model_update=1e-2,
                policy=BoltzmannQPolicy(), test_policy=BoltzmannQPolicy())
agent_dqn.compile(Adam(lr=1e-3), metrics=['mae'])

env_gym.env_dqn = env_dqn
env_gym.agent_dqn = agent_dqn


nround = 2000
result_folder = 'result/test/' #datetime.datetime.now().isoformat()
makedirs(result_folder)

for round in range(nround):
    print '\n\nround ' + str(round) + '/' + str(nround)

    print '\n\nagent '
    agent_dqn.fit(agent_gym, nb_steps=50000, nb_max_episode_steps=config.GeneratorEnv.MaxGameStep, visualize=True, verbose=2)

    #print '\n\nenv '
    #env_dqn.fit(env_gym, nb_steps=500, visualize=True, verbose=2)

    agent_dqn.save_weights(result_folder + '/agent_model_weights_{}.h5f'.format(str(round)), overwrite=True)
    env_dqn.save_weights(result_folder + '/generator_model_weights_{}.h5f'.format(str(round)), overwrite=True)


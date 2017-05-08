import config
from utils import *
import datetime
import numpy as np
from env_gym import ENV_GYM
from agent_gym import ADVERSARIAL_AGENT_GYM
from keras.optimizers import Adam, SGD
from rl.core import Processor
from rl.agents.dqn import DQNAgent as DQN
from rl.agents.ddpg import DDPGAgent as PG
from rl.random import OrnsteinUhlenbeckProcess
from policy import *
from rl.memory import SequentialMemory
from agent_net import get_agent_net
from env_net import *

np.random.seed(config.Game.Seed)

env_gym = ENV_GYM()
env_gym.seed(config.Game.Seed)

#env_actor = get_env_actor()
#env_critic, action_input = get_env_critic()
#random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0, sigma=.3, size=env_gym.action_space.n)
#env = PG(actor=env_actor, critic=env_critic, critic_action_input=action_input, gamma=1.0, nb_actions=env_gym.action_space.n, memory=env_memory, nb_steps_warmup_actor=50, nb_steps_warmup_critic=50, target_model_update=1e-3, random_process=random_process)

env_net = get_env_net()
env_memory = SequentialMemory(limit=50000, window_length=1)

#env_policy = EpsABCPolicy(policyA=MaskedGreedyQPolicy(), policyB=MaskedBoltzmannQPolicy(), policyC=MaskedRandomPolicy(), eps_forB=0.1, eps_forC=0.01)
env_policy = EpsABPolicy(policyA=MaskedGreedyQPolicy(), policyB=MaskedRandomPolicy(), eps_forB=0.1)
env_test_policy = MaskedBoltzmannQPolicy()

env = DQN(model=env_net, gamma=1.0, nb_actions=env_gym.action_space.n, memory=env_memory, nb_steps_warmup=100, target_model_update=1e-3, enable_dueling_network=False, policy=env_policy, test_policy=env_test_policy)
env.compile(Adam(lr=1e-3), metrics=['mae'])

agent_env_policy = EpsABPolicy(policyA=MaskedGreedyQPolicy(), policyB=MaskedRandomPolicy(),eps_forB=0.1)
agent_gym = ADVERSARIAL_AGENT_GYM(env_gym, env_test_policy)
agent_gym.seed(config.Game.Seed)

agent_net = get_agent_net()
agent_memory = SequentialMemory(limit=50000, window_length=1)

agent_policy = EpsABPolicy(policyA=GreedyQPolicy(), policyB=RandomPolicy(), eps_forB=0.1)
agent_test_policy = GreedyQPolicy()

agent = DQN(model=agent_net, gamma=1.0, nb_actions=agent_gym.action_space.n, memory=agent_memory, nb_steps_warmup=100, target_model_update=1e-3, enable_dueling_network=True, policy=agent_policy, test_policy=agent_test_policy)
agent.compile(Adam(lr=1e-3), metrics=['mae'])

env_gym.env = env
env_gym.agent = agent


nround = 2000
result_folder = 'result/test/' #datetime.datetime.now().isoformat()
makedirs(result_folder)

for round in range(nround):
    print('\n\nround ' + str(round) + '/' + str(nround))

    #print '\n\nagent '
    #agent.fit(agent_gym, nb_steps=1000, nb_max_episode_steps=config.Game.MaxGameStep, visualize=False, verbose=2)
    #agent.test(agent_gym, nb_episodes=10, nb_max_episode_steps=config.Game.MaxGameStep, visualize=False, verbose=2)

    print('\n\nenv ')
    env.fit(env_gym, nb_steps=1000, visualize=False, verbose=2)
    env.test(env_gym, nb_episodes=10, visualize=False, verbose=2)

    #agent.save_weights(result_folder + '/agent_model_weights_{}.h5f'.format(str(round)), overwrite=True)
    #env.save_weights(result_folder + '/generator_model_weights_{}.h5f'.format(str(round)), overwrite=True)


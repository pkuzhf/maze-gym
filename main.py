import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import config
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
from rl.memory import SequentialMemory

import keras.backend.tensorflow_backend as KTF
KTF.set_session(get_session())

def main():
    
    print len(sys.argv)
    print sys.argv

    if len(sys.argv) >= 2:
        task_name = sys.argv[1]
    else:
        task_name = 'default'

    if len(sys.argv) >= 4:
        config.Map.Height = int(sys.argv[2])
        config.Map.Width = int(sys.argv[3])

    np.random.seed(config.Game.Seed)

    env_gym = ENV_GYM()
    env_gym.seed(config.Game.Seed)

    env_net = get_env_net()
    env_memory = SequentialMemory(limit=50000, window_length=1)

    env_s = config.Training.RewardScale # the significant reward scale
    env_tau = get_tau(env_s)
    env_policy = EpsABPolicy(policyA=MaskedBoltzmannQPolicy(tau=env_tau), policyB=MaskedRandomPolicy(), eps_forB=config.Training.EnvTrainEpsForB, half_eps_step=1000, eps_min=0.1)
    env_test_policy = MaskedBoltzmannQPolicy(tau=env_tau)

    env = DQN(model=env_net, gamma=1.0, nb_steps_warmup=config.Training.EnvWarmup, target_model_update=config.Training.EnvTargetModelUpdate, 
        enable_dueling_network=False, policy=env_policy, test_policy=env_test_policy, nb_actions=env_gym.action_space.n, memory=env_memory)
    env.compile(Adam(lr=config.Training.EnvLearningRate))

    agent_env_policy = EpsABPolicy(policyA=MaskedBoltzmannQPolicy(tau=env_tau), policyB=MaskedRandomPolicy(), eps_forB=0.1)
    agent_gym = ADVERSARIAL_AGENT_GYM(env_gym, agent_env_policy)
    agent_gym.seed(config.Game.Seed)

    agent_net = get_agent_net()
    agent_memory = SequentialMemory(limit=50000, window_length=1)

    agent_s = config.Training.RewardScale # the significant reward scale
    agent_tau = get_tau(agent_s)
    agent_policy = EpsABPolicy(policyA=GreedyQPolicy(), policyB=RandomPolicy(), eps_forB=config.Training.AgentTrainEpsForB, 
        half_eps_step=1000, eps_min=0.1)
    agent_test_policy = EpsABPolicy(policyA=GreedyQPolicy(), policyB=RandomPolicy(), eps_forB=config.Training.AgentTestEpsForB)

    agent = DQN(model=agent_net, gamma=1.0, nb_steps_warmup=config.Training.AgentWarmup, target_model_update=config.Training.AgentTargetModelUpdate,
     enable_dueling_network=False, policy=agent_policy, test_policy=agent_test_policy, nb_actions=agent_gym.action_space.n, memory=agent_memory)
    agent.compile(Adam(lr=config.Training.AgentLearningRate))

    env_gym.env = env
    env_gym.agent = agent

    nround = 1000
    result_folder = '../maze_result' #datetime.datetime.now().isoformat()
    makedirs(result_folder)

    print vars(config.Map)
    print vars(config.Training)

    for round in range(nround):
        print('\n\nround ' + str(round) + '/' + str(nround))

        print('\n\nagent')
        agent.fit(agent_gym, nb_steps=500 if round>5 else 5000, nb_max_episode_steps=config.Game.MaxGameStep, visualize=False, verbose=2)
        agent.test(agent_gym, nb_episodes=10, nb_max_episode_steps=config.Game.MaxGameStep, visualize=False, verbose=2)

        print('\n\nenv')
        env.fit(env_gym, nb_steps=500, visualize=False, verbose=2)
        env.test(env_gym, nb_episodes=10, visualize=False, verbose=2)

        agent.save_weights(result_folder + '/{}_agent_model_weights_{}.h5f'.format(task_name, str(round)), overwrite=True)
        env.save_weights(result_folder + '/{}_generator_model_weights_{}.h5f'.format(task_name, str(round)), overwrite=True)

if __name__ == "__main__":
    main()
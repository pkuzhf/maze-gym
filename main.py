import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
from rl.memory import SequentialMemory

import keras.backend.tensorflow_backend as KTF
KTF.set_session(get_session())

def main():

    argv = '\n\n'
    for arg in sys.argv:
        argv += arg + ' '
    print(argv)

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
    env_memory = SequentialMemory(limit=1000, window_length=1)
    #BoltzmannQPolicy(tau=get_tau(config.Training.RewardScaleTrain))
    env_policy = EpsABPolicy(policyA=GreedyQPolicy(), policyB=RandomPolicy(),
        eps_forB=config.Training.EnvTrainEps, half_eps_step=config.Training.EnvTrainEps_HalfStep, eps_min=config.Training.EnvTrainEps_Min)
    env_test_policy = BoltzmannQPolicy(tau=get_tau(config.Training.RewardScaleTest))

    env = mDQN(name='env', model=env_net, gamma=1.0, nb_steps_warmup=config.Training.EnvWarmup, target_model_update=config.Training.EnvTargetModelUpdate,
        enable_dueling_network=True, policy=env_policy, test_policy=env_test_policy, nb_actions=env_gym.action_space.n, memory=env_memory)
    env.compile(Adam(lr=config.Training.EnvLearningRate))

    agent_env_policy = env_policy
    #EpsABPolicy(policyA=BoltzmannQPolicy(tau=get_tau(config.Training.RewardScaleGen)), policyB=RandomPolicy(), eps_forB=config.Training.EnvEpsGen)
    agent_gym = ADVERSARIAL_AGENT_GYM(env_gym, agent_env_policy)
    agent_gym.seed(config.Game.Seed)

    agent_net = get_agent_net()
    agent_memory = SequentialMemory(limit=1000, window_length=1)

    agent_policy = EpsABPolicy(policyA=GreedyQPolicy(), policyB=RandomPolicy(), eps_forB=config.Training.AgentTrainEps,
        half_eps_step=config.Training.AgentTrainEps_HalfStep, eps_min=config.Training.AgentTrainEps_Min)
    agent_test_policy = GreedyQPolicy()

    agent = mDQN(name='agent', model=agent_net, gamma=1.0, nb_steps_warmup=config.Training.AgentWarmup, target_model_update=config.Training.AgentTargetModelUpdate,
        enable_dueling_network=True, policy=agent_policy, test_policy=agent_test_policy, nb_actions=agent_gym.action_space.n, memory=agent_memory)
    agent.compile(Adam(lr=config.Training.AgentLearningRate))

    env_gym.env = env
    env_gym.agent = agent
    agent_gym.agent = agent

    print(argv)
    print(vars(config.Map))
    print(vars(config.Training))

    #run(agent, env, agent_gym, env_gym, task_name)
    run_env_path(env, env_gym, task_name)

    print(argv)
    print(vars(config.Map))
    print(vars(config.Training))


def run_env_path(env, env_gym, task_name):

    nround = 1000
    model_folder = config.Path.Models
    makedirs(model_folder)

    for round in range(nround):

        print('\n\nround train' + str(round) + '/' + str(nround))
        env.fit(env_gym, nb_episodes=100, min_steps=100, visualize=False, verbose=2)
        env.nb_steps_warmup = 0
        env.test(env_gym, nb_episodes=10, visualize=False, verbose=2)
        env.save_weights(model_folder + '/{}_generator_model_weights_{}.h5f'.format(task_name, str(round)), overwrite=True)


def run(agent, env, agent_gym, env_gym, task_name):

    nround = 1000
    model_folder = config.Path.Models
    makedirs(model_folder)

    for round in range(nround):

        print('\n\nround train' + str(round) + '/' + str(nround))

        for subround in range(100):

            #print('\n\nagent: subround ' + str(round) + ' / ' + str(subround))
            #agent.fit(agent_gym, nb_episodes=10, min_steps=100, nb_max_episode_steps=config.Game.MaxGameStep, visualize=False, verbose=2)
            #agent.nb_steps_warmup = 0
            #agent.test(agent_gym, nb_episodes=1, nb_max_episode_steps=config.Game.MaxGameStep, visualize=False, verbose=2)

            print('\n\nenv: subround ' + str(round) + ' / ' + str(subround))
            env.fit(env_gym, nb_episodes=10, min_steps=100, visualize=False, verbose=2)
            env.nb_steps_warmup = 0
            env.test(env_gym, nb_episodes=1, visualize=False, verbose=2)

        print('\n\nround test' + str(round) + '/' + str(nround))
        #agent.test(agent_gym, nb_episodes=10, nb_max_episode_steps=config.Game.MaxGameStep, visualize=False, verbose=2)
        env.test(env_gym, nb_episodes=10, visualize=False, verbose=2)

        print('\n\nround save' + str(round) + '/' + str(nround))
        #agent.save_weights(model_folder + '/{}_agent_model_weights_{}.h5f'.format(task_name, str(round)), overwrite=True)
        env.save_weights(model_folder + '/{}_generator_model_weights_{}.h5f'.format(task_name, str(round)), overwrite=True)


if __name__ == "__main__":
    main()
    #profile.run("main()", sort=1)


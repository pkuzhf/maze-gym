Adversarial Environment Experiments

Our experiments depend on Keras-RL(https://github.com/matthiasplappert/keras-rl), Tensorflow(https://www.tensorflow.org/) and OpenAI Gym(https://gym.openai.com/). Please install these dependencies before running our experiment codes.

1. Test transition gradient on soft wall maze

Run "python transition_gradient.py DQN 5 5" and "python transition_gradient.py OPT 5 5". 

Parameters can be set in the main function in this file.

2. Test generative model on maze

Run "python main.py %task_name %height %width", e.g. "python main.py shortest 5 5".

%task_name includes "shortest", "dfs", "right_hand" and "dqn". 
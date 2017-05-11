from __future__ import division
import warnings
import numpy as np
from copy import deepcopy
from keras.callbacks import History
from rl.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList
from rl.agents.dqn import DQNAgent
from myCallback import myTrainEpisodeLogger

class myDQNAgent(DQNAgent):

    def fit(self, env, nb_episodes, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        if not self.compiled:
            raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [myTrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        #callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        self.step = 0
        observation = None
        episode_reward = None
        episode_step = None

        for episode in range(nb_episodes):

            callbacks.on_episode_begin(episode)
            episode_step = 0
            episode_reward = 0.
            self.reset_states()
            observation = deepcopy(env.reset())

            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            #nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
            #for _ in range(nb_random_start_steps):
            #    if start_step_policy is None:
            #        action = env.action_space.sample()
            #    else:
            #        action = start_step_policy(observation)
            #    callbacks.on_action_begin(action)
            #    observation, reward, done, info = env.step(action)
            #    observation = deepcopy(observation)
            #    if self.processor is not None:
            #        observation, reward, done, info = self.processor.process_step(observation, reward, done, info)
            #    callbacks.on_action_end(action)
            #    if done:
            #        warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
            #        observation = deepcopy(env.reset())
            #        if self.processor is not None:
            #            observation = self.processor.process_observation(observation)
            #        break
            # At this point, we expect to be fully initialized.

            done = False
            while not done:

                callbacks.on_step_begin(episode_step)
                action = self.forward(observation)

                callbacks.on_action_begin(action)
                observation, reward, done, info = env.step(action)
                observation = deepcopy(observation)

                callbacks.on_action_end(action)

                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True

                metrics = self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': episode,
                }
                callbacks.on_step_end(episode_step, step_logs)

                episode_step += 1
                self.step += 1

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.

                    self.forward(observation)
                    self.backward(0., terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

        callbacks.on_train_end(logs={'did_abort': False})
        self._on_train_end()

        return history

    def forward(self, observation):

        q_values = self.compute_q_values([observation]) # only for windows 1
        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)

        self.recent_observation = observation
        self.recent_action = action

        return action
from __future__ import division
import warnings
import numpy as np
import warnings
from copy import deepcopy

import keras.backend as K
from keras.layers import Lambda, Input, Layer, Dense
from rl.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList
from keras.callbacks import History

from rl.core import Agent
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.util import *
from rl.keras_future import Model

from rl.agents.dqn import DQNAgent
from myCallback import myTrainEpisodeLogger
from collections import deque
from utils import qlogger, displayQvalue

class myDQNAgent(DQNAgent):

    def __init__(self, name='', *args, **kwargs):

        self.name = name
        super(myDQNAgent, self).__init__(*args, **kwargs)

        self.max_reward = -1e20
        self.reward_his = deque(maxlen=10000)
        self.total_step = 0

        self.qlogger = qlogger()
        self.policy.qlogger = self.qlogger
        self.test_policy.qlogger = self.qlogger

    def fit(self, env, nb_episodes, min_steps, action_repetition=1, callbacks=None, verbose=1,
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
            callbacks += [myTrainEpisodeLogger(self)]
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
            'name': self.name,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        self.step = 0

        episode = 0
        while episode<nb_episodes or self.step < min_steps:

            callbacks.on_episode_begin(episode)
            episode_step = 0
            episode_reward = 0.
            self.reset_states()
            observation = deepcopy(env.reset())

            while True:

                callbacks.on_step_begin(episode_step)

                q_values = self.compute_q_values([observation])  # only for windows 1
                action = self.policy.select_action(q_values=q_values)

                self.recent_observation = observation
                self.recent_action = action

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
                    self.policy.log_qvalue(q_values)
                    cur_maxq = self.qlogger.cur_maxq
                    if self.policy.mask is not None:
                        displayQvalue(q_values)#*(1-self.policy.mask))
                    else:
                        displayQvalue(q_values)

                    self.forward(observation)
                    self.backward(0., terminal=False)
                    break

            episode_logs = {
                'episode_reward': episode_reward,
                'nb_episode_steps': episode_step,
                'nb_steps': self.step,
                'q_value': q_values[action],
                'q_max': cur_maxq,
                'q_mean': np.mean(self.qlogger.mean_maxq)
            }
            callbacks.on_episode_end(episode, episode_logs)
            episode += 1

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

    def compile(self, optimizer, metrics=None):

        metrics = []

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_masked_error(args):
            y_true, y_pred, mask = args
            loss = huber_loss(y_true, y_pred, self.delta_clip)
            loss *= mask  # apply element-wise mask
            return K.sum(loss, axis=-1)

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        y_pred = self.model.output
        y_true = Input(name='y_true', shape=(self.nb_actions,))
        mask = Input(name='mask', shape=(self.nb_actions,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_pred, y_true, mask])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        trainable_model = Model(input=ins + [y_true, mask], output=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

        self.compiled = True

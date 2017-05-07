from __future__ import division
from rl.util import *

class Policy(object):

    def __init__(self):
        self.mask = None

    def _set_agent(self, agent):
        self.agent = agent

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    def select_action(self, **kwargs):
        raise NotImplementedError()

    def get_config(self):
        return {}


class EpsGreedyQPolicy(Policy):
    def __init__(self, eps=.1):
        super(EpsGreedyQPolicy, self).__init__()
        self.eps = eps

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.random_integers(0, nb_actions-1)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config


class GreedyQPolicy(Policy):
    def select_action(self, q_values):
        assert q_values.ndim == 1
        action = np.argmax(q_values)
        return action


class BoltzmannQPolicy(Policy):
    def __init__(self, tau=1.):
        super(BoltzmannQPolicy, self).__init__()
        self.tau = tau
        self.minq = 1e20
        self.maxq = -1e20

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        q_values = q_values.astype('float64')

        if np.isnan(q_values).any():
            print q_values
        if self.minq > np.min(q_values):
            self.minq = np.min(q_values)
            print self.minq, self.maxq
        if self.maxq < np.max(q_values):
            self.maxq = np.max(q_values)
            print self.minq, self.maxq

        q_values /= self.tau
        q_values -= np.max(q_values)
        exp_values = np.exp(q_values)
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_config(self):
        config = super(BoltzmannQPolicy, self).get_config()
        return config


class EpsBoltzmannQPolicy(Policy):
    def __init__(self, eps=.1, tau=1.):
        super(EpsBoltzmannQPolicy, self).__init__()
        self.eps = eps
        self.tau = tau

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        if np.random.uniform() < self.eps:
            action = np.random.random_integers(0, nb_actions - 1)
        else:
            q_values = q_values.astype('float64')
            q_values /= self.tau
            q_values -= np.max(q_values)
            exp_values = np.exp(q_values)
            probs = exp_values / np.sum(exp_values)
            action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_config(self):
        config = super(EpsBoltzmannQPolicy, self).get_config()
        config['eps'] = self.eps
        return config


class MaskedBoltzmannQPolicy(Policy):
    def __init__(self, tau=1.):
        super(MaskedBoltzmannQPolicy, self).__init__()
        self.minq = 1e20
        self.maxq = -1e20
        self.tau = tau
        self.mask = None

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        q_values = q_values.astype('float64')

        if np.isnan(q_values).any():
            print q_values
        if self.minq > np.min(q_values):
            self.minq = np.min(q_values)
            print self.minq, self.maxq
        if self.maxq < np.max(q_values):
            self.maxq = np.max(q_values)
            print self.minq, self.maxq

        q_values /= self.tau
        q_values -= np.max(q_values)
        if self.mask is not None:
            q_values -= self.mask * 1e20

        exp_values = np.exp(q_values)
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_config(self):
        config = super(MaskedBoltzmannQPolicy, self).get_config()
        return config

    def set_mask(self, mask):
        self.mask = mask


class MaskedEpsGreedyQPolicy(Policy):
    def __init__(self, eps=.1):
        super(MaskedEpsGreedyQPolicy, self).__init__()
        self.eps = eps
        self.mask = None

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            probs = np.ones(nb_actions)
            if self.mask is not None:
                probs -= self.mask
            probs /= np.sum(probs)
            action = np.random.choice(range(nb_actions), p=probs)
        else:
            if self.mask is not None:
                q_values -= self.mask * 1e20
            action = np.argmax(q_values)

        return action

    def get_config(self):
        config = super(MaskedEpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config

    def set_mask(self, mask):
        self.mask = mask


class MaskedGreedyQPolicy(Policy):

    def __init__(self):
        super(MaskedGreedyQPolicy, self).__init__()
        self.mask = None

    def select_action(self, q_values):
        assert q_values.ndim == 1
        if self.mask is not None:
            q_values -= self.mask * 1e20
        action = np.argmax(q_values)
        return action

    def set_mask(self, mask):
        self.mask = mask


class MaskedEpsSoftmaxQPolicy(Policy):
    def __init__(self, eps=.1):
        super(MaskedEpsSoftmaxQPolicy, self).__init__()
        self.eps = eps
        self.mask = None

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            probs  = np.ones(nb_actions)
            probs -= self.mask
            probs /= np.sum(probs)
            action = np.random.choice(range(nb_actions), p=probs)
        else:
            if self.mask is not None:
                q_values -= self.mask * 1e20
            action = np.argmax(q_values)
        return action

    def get_config(self):
        config = super(MaskedEpsSoftmaxQPolicy, self).get_config()
        config['eps'] = self.eps
        return config

    def set_mask(self, mask):
        self.mask = mask


class LinearAnnealedPolicy(Policy):
    def __init__(self, inner_policy, attr, value_max, value_min, value_test, nb_steps):
        if not hasattr(inner_policy, attr):
            raise ValueError('Policy "{}" does not have attribute "{}".'.format(attr))

        super(LinearAnnealedPolicy, self).__init__()

        self.inner_policy = inner_policy
        self.attr = attr
        self.value_max = value_max
        self.value_min = value_min
        self.value_test = value_test
        self.nb_steps = nb_steps

    def get_current_value(self):
        if self.agent.training:
            # Linear annealed: f(x) = ax + b.
            a = -float(self.value_max - self.value_min) / float(self.nb_steps)
            b = float(self.value_max)
            value = max(self.value_min, a * float(self.agent.step) + b)
        else:
            value = self.value_test
        return value

    def select_action(self, **kwargs):
        setattr(self.inner_policy, self.attr, self.get_current_value())
        return self.inner_policy.select_action(**kwargs)

    @property
    def metrics_names(self):
        return ['mean_{}'.format(self.attr)]

    @property
    def metrics(self):
        return [getattr(self.inner_policy, self.attr)]

    def get_config(self):
        config = super(LinearAnnealedPolicy, self).get_config()
        config['attr'] = self.attr
        config['value_max'] = self.value_max
        config['value_min'] = self.value_min
        config['value_test'] = self.value_test
        config['nb_steps'] = self.nb_steps
        config['inner_policy'] = get_object_config(self.inner_policy)
        return config


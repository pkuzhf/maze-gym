from __future__ import division
from rl.util import *

class Policy(object):

    def __init__(self):
        self.minq = 1e20
        self.maxq = -1e20
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
    
    def log_qvalue(self, q_values):
        if np.isnan(q_values).any():
            print(q_values)
        if self.minq > np.min(q_values):
            self.minq = np.min(q_values)
            #print(q_values)
        if self.maxq < np.max(q_values):
            self.maxq = np.max(q_values)
            #print(q_values)


class RandomPolicy(Policy):

    def select_action(self, q_values):
        self.log_qvalue(q_values)
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        action = np.random.random_integers(0, nb_actions - 1)
        return action


class BoltzmannQPolicy(Policy):

    def __init__(self, tau=1.):
        super(BoltzmannQPolicy, self).__init__()
        self.tau = tau

    def select_action(self, q_values):
        self.log_qvalue(q_values)
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        q_values = q_values.astype('float64')
        q_values /= self.tau
        q_values -= np.max(q_values)
        exp_values = np.exp(q_values)
        sum_exp = np.sum(exp_values)
        assert sum_exp >= 1.0
        probs = exp_values / sum_exp
        action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_config(self):
        config = super(BoltzmannQPolicy, self).get_config()
        return config


class GreedyQPolicy(Policy):

    def select_action(self, q_values):
        self.log_qvalue(q_values)
        assert q_values.ndim == 1
        action = np.argmax(q_values)
        return action


class MaskedRandomPolicy(Policy):

    def __init__(self):
        super(MaskedRandomPolicy, self).__init__()
        self.mask = None

    def select_action(self, q_values):
        self.log_qvalue(q_values)
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        probs = np.ones(nb_actions)
        if self.mask is not None:
            probs -= self.mask
        sum_probs = np.sum(probs)
        assert sum_probs >= 1.0
        probs /= sum_probs
        action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_config(self):
        config = super(MaskedRandomPolicy, self).get_config()
        return config

    def set_mask(self, mask):
        self.mask = mask


class MaskedBoltzmannQPolicy(Policy):

    def __init__(self, tau=1.):
        super(MaskedBoltzmannQPolicy, self).__init__()
        self.minq = 1e20
        self.maxq = -1e20
        self.tau = tau
        self.mask = None

    def select_action(self, q_values):
        self.log_qvalue(q_values)
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        q_values = q_values.astype('float64')
        if self.mask is not None:
            q_values -= self.mask * 1e20
        q_values /= self.tau
        q_values -= np.max(q_values)
        exp_values = np.exp(q_values)
        sum_exp = np.sum(exp_values)
        assert sum_exp >= 1.0
        probs = exp_values / sum_exp
        action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_config(self):
        config = super(MaskedBoltzmannQPolicy, self).get_config()
        return config

    def set_mask(self, mask):
        self.mask = mask


class MaskedGreedyQPolicy(Policy):

    def __init__(self):
        super(MaskedGreedyQPolicy, self).__init__()
        self.mask = None

    def select_action(self, q_values):
        self.log_qvalue(q_values)
        assert q_values.ndim == 1
        if self.mask is not None:
            q_values -= self.mask * 1e20
        action = np.argmax(q_values)
        return action

    def set_mask(self, mask):
        self.mask = mask


class EpsABPolicy(Policy):

    def __init__(self, policyA, policyB, eps_forB, eps_decay_rate_each_step=1.0):
        super(EpsABPolicy, self).__init__()
        self.policyA = policyA
        self.policyB = policyB
        self.eps_forB = eps_forB
        self.eps_decay_rate_each_step = eps_decay_rate_each_step

    def select_action(self, q_values):
        self.log_qvalue(q_values)
        assert q_values.ndim == 1
        if np.random.uniform() < self.eps_forB:
            action = self.policyB.select_action(q_values)
        else:
            action = self.policyA.select_action(q_values)
        self.eps_forB *= self.eps_decay_rate_each_step
        return action

    def get_config(self):
        config = super(EpsABPolicy, self).get_config()
        config['policyA'] = self.policyA
        config['policyB'] = self.policyB
        config['eps_forB'] = self.eps_forB
        config['eps_decay_rate_each_step'] = self.eps_decay_rate_each_step
        return config

    def set_mask(self, mask):
        self.mask = mask
        self.policyA.set_mask(self.mask)
        self.policyB.set_mask(self.mask)


class EpsABCPolicy(Policy):
    def __init__(self, policyA, policyB, policyC, eps_forB, eps_forC, eps_decay_rate_each_step=1.0):
        super(EpsABCPolicy, self).__init__()
        self.policyA = policyA
        self.policyB = policyB
        self.policyC = policyC
        self.eps_forB = eps_forB
        self.eps_forC = eps_forC
        self.eps_decay_rate_each_step = eps_decay_rate_each_step

    def select_action(self, q_values):
        self.log_qvalue(q_values)
        assert q_values.ndim == 1
        rand = np.random.uniform()
        if rand < self.eps_forC:
            action = self.policyC.select_action(q_values)
        elif rand < self.eps_forC + self.eps_forB:
            action = self.policyB.select_action(q_values)
        else:
            action = self.policyA.select_action(q_values)
        self.eps_forB *= self.eps_decay_rate_each_step
        self.eps_forC *= self.eps_decay_rate_each_step
        return action

    def get_config(self):
        config = super(EpsABCPolicy, self).get_config()
        config['policyA'] = self.policyA
        config['policyB'] = self.policyB
        config['policyC'] = self.policyC
        config['eps_forB'] = self.eps_forB
        config['eps_decay_rate_each_step'] = self.eps_decay_rate_each_step
        return config

    def set_mask(self, mask):
        self.mask = mask
        self.policyA.set_mask(self.mask)
        self.policyB.set_mask(self.mask)
        self.policyC.set_mask(self.mask)

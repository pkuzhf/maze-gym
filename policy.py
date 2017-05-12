from __future__ import division
from rl.util import *

class Policy(object):

    def __init__(self):
        self.mask = None
        self.qlogger = None
        self.eps_forB = 0
        self.eps_forC = 0

    def _set_agent(self, agent):
        self.agent = agent

    def set_mask(self, mask):
        self.mask = mask

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

        if self.qlogger is not None:

            if self.mask is not None:
                q_values = q_values - self.mask * 1e20

            self.qlogger.pre_maxq = self.qlogger.cur_maxq

            self.qlogger.cur_maxq = np.max(q_values)

            if self.qlogger.maxq < self.qlogger.cur_maxq:
                self.qlogger.maxq = self.qlogger.cur_maxq

            self.qlogger.mean_maxq.append(self.qlogger.cur_maxq)

class RandomPolicy(Policy):

    def select_action(self, q_values):
        #self.log_qvalue(q_values)
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        action = np.random.random_integers(0, nb_actions - 1)
        return action


class BoltzmannQPolicy(Policy):

    def __init__(self, tau=1.):
        super(BoltzmannQPolicy, self).__init__()
        self.tau = tau

    def select_action(self, q_values):
        #self.log_qvalue(q_values)
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
        #self.log_qvalue(q_values)
        assert q_values.ndim == 1
        action = np.argmax(q_values)
        return action


class MaskedRandomPolicy(Policy):

    def __init__(self):
        super(MaskedRandomPolicy, self).__init__()
        self.mask = None

    def select_action(self, q_values):
        #self.log_qvalue(q_values)
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


class MaskedBoltzmannQPolicy(Policy):

    def __init__(self, tau=1.):
        super(MaskedBoltzmannQPolicy, self).__init__()
        self.minq = 1e20
        self.maxq = -1e20
        self.tau = tau
        self.mask = None

    def select_action(self, q_values):
        #self.log_qvalue(q_values)
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


class MaskedGreedyQPolicy(Policy):

    def __init__(self):
        super(MaskedGreedyQPolicy, self).__init__()
        self.mask = None

    def select_action(self, q_values):
        #self.log_qvalue(q_values)
        assert q_values.ndim == 1
        if self.mask is not None:
            q_values -= self.mask * 1e20
        action = np.argmax(q_values)
        return action


class EpsABPolicy(Policy):

    def __init__(self, policyA, policyB, eps_forB, half_eps_step=0, eps_min=0):
        super(EpsABPolicy, self).__init__()
        self.policyA = policyA
        self.policyB = policyB
        self.eps_forB = eps_forB
        self.eps_min=eps_min
        if half_eps_step==0:
            self.eps_decay_rate_each_step = 1.0
        else:
            self.eps_decay_rate_each_step = np.power(0.5, 1.0/half_eps_step)

    def select_action(self, q_values):
        #self.log_qvalue(q_values)
        assert q_values.ndim == 1
        if np.random.uniform() < self.eps_forB:
            action = self.policyB.select_action(q_values)
        else:
            action = self.policyA.select_action(q_values)
        self.eps_forB *= self.eps_decay_rate_each_step
        self.eps_forB = max(self.eps_min, self.eps_forB)
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
    def __init__(self, policyA, policyB, policyC, eps_forB, eps_forC, half_eps_step=0, eps_min=0):
        super(EpsABCPolicy, self).__init__()
        self.policyA = policyA
        self.policyB = policyB
        self.policyC = policyC
        self.eps_forB = eps_forB
        self.eps_forC = eps_forC
        self.eps_min = eps_min
        if half_eps_step == 0:
            self.eps_decay_rate_each_step = 1.0
        else:
            self.eps_decay_rate_each_step = np.power(0.5, 1.0 / half_eps_step)

    def select_action(self, q_values):
        #self.log_qvalue(q_values)
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
        self.eps_forB = max(self.eps_min, self.eps_forB)
        self.eps_forC = max(self.eps_min, self.eps_forC)
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

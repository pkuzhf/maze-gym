
from rl.callbacks import *
from collections import deque

class myTrainEpisodeLogger(Callback):

    def __init__(self, dqn):
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        self.episode_start = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.step = 0
        self.dqn = dqn

    def on_train_begin(self, logs):
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        print('Training for {} episodes ...'.format(self.params['nb_episodes']))

    def on_train_end(self, logs):
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))

    def on_episode_begin(self, episode, logs):
        self.episode_start[episode] = timeit.default_timer()
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []

    def on_episode_end(self, episode, logs):
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.rewards[episode])

        metrics = np.array(self.metrics[episode])
        metrics_template = ''
        metrics_variables = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                if idx > 0:
                    metrics_template += ', '
                try:
                    value = np.nanmean(metrics[:, idx])
                    metrics_template += '{}: {:.3f}'
                except Warning:
                    value = '--'
                    metrics_template += '{}: {}'
                metrics_variables += [name, value]
        metrics_text = metrics_template.format(*metrics_variables)

        episode_reward = np.sum(self.rewards[episode])

        template = '{name} episode: {episode}, step: {episode_steps}, ' \
                   'max reward: {max_reward:.2f}, avg reward: {average_reward_his:.2f}, cur reward {episode_reward:.2f}, '\
                   'cur qvalue: {curq:.3f}, max qvalue: {cur_maxq:.3f}, avg qvalue: {mean_maxq:.3f}, eps: {eps:.3f}, ' \
                   'steps per second: {sps:.1f}, duration: {duration:.3f}s, {metrics}, '\
                   'mean reward: {reward_mean:.3f} [{reward_min:.3f}, {reward_max:.3f}], mean action: {action_mean:.3f} [{action_min:.3f}, {action_max:.3f}], ' \
                   'episode step: {episode_step}, total step: {total_step}, maxm reward: {maxm_reward:.2f},'

        variables = {
            'name': self.params['name'],
            'episode': episode + 1,
            'episode_step': self.step,
            'total_step': self.dqn.total_step,
            'duration': duration,
            'episode_steps': episode_steps,
            'eps': self.dqn.policy.eps_forB,
            'sps': float(episode_steps) / duration,
            'episode_reward': episode_reward,
            'average_reward_his': np.mean(self.dqn.reward_his),
            'max_reward': np.max(self.dqn.reward_his),
            'maxm_reward': self.dqn.max_reward,
            'metrics': metrics_text,
            'curq': logs['q_value'], #self.dqn.qlogger.pre_minq,
            'cur_maxq': logs['q_max'], #self.dqn.qlogger.pre_maxq,
            'mean_maxq': logs['q_mean'],
            'reward_mean': np.mean(self.rewards[episode]),
            'reward_min': np.min(self.rewards[episode]),
            'reward_max': np.max(self.rewards[episode]),
            'action_mean': np.mean(self.actions[episode]),
            'action_min': np.min(self.actions[episode]),
            'action_max': np.max(self.actions[episode])
        }
        print(template.format(**variables))

        # Free up resources.
        del self.episode_start[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.step += 1
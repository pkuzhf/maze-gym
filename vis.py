# coding: utf-8

# In[139]:

# %matplotlib inline
import matplotlib

matplotlib.use('Agg')
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot  as plt
import os
import glob
import seaborn as sns
from collections import deque


# In[140]:

def plot(log_path):
    procesed_log_path = log_path.replace('.log', '.processed.log')
    fig_path = log_path.replace('.log', '.pdf')
    is_env = True
    test_log_begin = False
    is_map_begin = False
    procesed_log = {'Environment': [], 'Agent': []}
    global_env_test_times = 0
    global_agent_test_times = 0
    sub_test_times = 0
    map_string = ''

    with open(log_path, 'r') as fo:
        for line in fo:
            if line.startswith('agent: subround '):
                is_env = False
            if line.startswith('env: subround '):
                is_env = True
            if line.startswith('Testing for '):
                if is_env:
                    global_env_test_times += 1
                if not is_env:
                    global_agent_test_times += 1
                sub_test_times = 0
                test_log_begin = True
            if test_log_begin is True:
                if line.startswith('Training for '):
                    test_log_begin = False
                    is_map_begin = False
                    continue
                if line.startswith('S'):
                    is_map_begin = True
                    #continue
                if is_map_begin is True:
                    if line.startswith('agent rewards'):
                        continue
                    map_string += line
                if line.startswith('Episode '):
                    is_map_begin = False
                    reward = float(line.split(',')[0].split(': ')[-1])
                    sub_test_times += 1
                    if is_env:
                        procesed_log['Environment'].append(
                            {'global_test_times': global_env_test_times, 'sub_test_times': sub_test_times,
                             'reward': reward, 'map_string': map_string})
                    if not is_env:
                        procesed_log['Agent'].append(
                            {'global_test_times': global_agent_test_times, 'sub_test_times': sub_test_times,
                             'reward': reward, 'map_string': map_string})
                    map_string = ''

    with open(procesed_log_path, 'w') as fout:
        json.dump(procesed_log, fout)

    if procesed_log['Environment'].__len__() == 0:
        return

    idx = -1
    while True:
        if procesed_log['Environment'][idx].has_key('global_test_times'):
            break
        else:
            idx -= 1

    max_test_times = procesed_log['Environment'][idx]['global_test_times']

    total_point = 100
    df = []
    skip_factor = int(max_test_times / total_point)
    if skip_factor < 1:
        skip_factor = 1

    type_reward = 2
    if type_reward == 0:
        accumulate_rewards = [0.] * 100
        for key in procesed_log.keys():
            logs = procesed_log[key]
            for log in logs:
                accumulate_rewards[log['sub_test_times']] += log['reward']
                if log['global_test_times'] % skip_factor == 0:
                    df.append([log['global_test_times'], key, log['sub_test_times'],
                               accumulate_rewards[log['sub_test_times']] / skip_factor])
                accumulate_rewards = [0.] * 100
    elif type_reward == 1:
        last_reward = 0
        for key in procesed_log.keys():
            logs = procesed_log[key]
            idx = 0
            while idx < len(logs):
                log = logs[idx]
                if log['global_test_times'] % skip_factor == 0:
                    if log['reward'] < 100:
                        last_reward = log['reward']
                        df.append([log['global_test_times'], key, log['sub_test_times'], log['reward']])
                    else:
                        df.append([log['global_test_times'], key, log['sub_test_times'], last_reward]) #trick
                idx += 1
    else:
        for key in procesed_log.keys():
            logs = procesed_log[key]
            idx = 0
            rewards = deque(maxlen=100)
            while idx < len(logs):
                if logs[idx]['reward'] < 100:
                    rewards.append(logs[idx]['reward'])
                else:
                    #rewards.append(logs[idx]['reward'])
                    pass
                if logs[idx]['global_test_times'] % skip_factor == 0 and (idx==len(logs)-1 or logs[idx+1]['global_test_times'] != logs[idx]['global_test_times']):
                    for r in range(len(rewards)):
                        df.append([logs[idx]['global_test_times'], key, r, rewards[r]])
                idx += 1

    df = pd.DataFrame(df)
    df.columns = ['Training times', 'Type', 'Sub training times', 'Reward']
    tmp = df[['Training times', 'Sub training times']].groupby(['Training times'])
    minimum_subtimes = tmp['Sub training times'].agg({'max': np.max})['max'].min()
    sns.plt.rcParams['figure.figsize'] = (10, 100)
    plt.rcParams['figure.figsize'] = (10, 1000)
    sns.set(style="whitegrid", font_scale=2)


    plot = sns.tsplot(data=df#[df['Sub training times'] <= minimum_subtimes]
                      , time="Training times",
                      unit="Sub training times",
                      condition="Type", value="Reward")

    plt.tight_layout()

    plot.get_figure().savefig(fig_path, format='pdf')
    plt.show()
    plot.clear()


# In[141]:
#plot('./logs/env_dqn_6x6.20170513_115332.log'); exit(0)

for fd in glob.glob("./logs/*.processed.log"):
    os.remove(fd)

paths = glob.glob('./logs/*.log')
for log_path in paths:
    print('working on:' + log_path)
    try:
        plot(log_path)
        print('done:'+log_path)
    except:
        print('error:'+log_path)


# In[88]:

# ### read
# log_data = json.load(open(procesed_log_path, 'r'))
# target_global_test_times = 1000
# target_log = []
# for log in log_data['Environment']:
#     if log['global_test_times'] == target_global_test_times:
#         target_log.append(log)


# In[ ]:




# In[ ]:

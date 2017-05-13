
# coding: utf-8

# In[42]:

# %matplotlib inline
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot  as plt
import glob
import seaborn as sns


# In[43]:

log_path = './logs/dfs.20170513_204108.log'
procesed_log_path = log_path.replace('.log', '.processed.log')
fig_path = log_path.replace('.log', '.pdf')
is_env = True
test_log_begin = False
is_map_begin = False
procesed_log = {'Environment':[], 'Agent':[]}


# In[44]:

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
            if line == '\n':
                test_log_begin = False
                is_map_begin = False
                continue
            if line.startswith('(\'env_step\''):
                is_map_begin = True
                continue
            if line.startswith('agent rewards'):
                is_map_begin = False
            if is_map_begin is True:
                map_string += line
            if line.startswith('Episode '):
                reward = float(line.split(',')[0].split(': ')[-1])
                sub_test_times += 1
                if is_env:
                    procesed_log['Environment'].append({'global_test_times': global_env_test_times, 'sub_test_times': sub_test_times,'reward': reward, 'map_string': map_string}) 
                if not is_env:
                    procesed_log['Agent'].append({'global_test_times': global_agent_test_times, 'sub_test_times': sub_test_times,'reward': reward, 'map_string': map_string}) 
                


# In[45]:

with open(procesed_log_path, 'w') as fout:
    json.dump(procesed_log, fout)


# In[57]:

df = []
for key in procesed_log.keys():
    logs = procesed_log[key]
    for log in logs:
        df.append([log['global_test_times'], key, log['sub_test_times'], log['reward']])
df = pd.DataFrame(df)    
df.columns = ['Training times', 'Type', 'Sub training times', 'Reward']


# In[58]:

df.head()


# In[59]:

sns.plt.rcParams['figure.figsize']=(10,10)
plt.rcParams['figure.figsize']=(10,10)
sns.set(style="whitegrid",font_scale=2)

plot = sns.tsplot(data=df, time="Training times", unit="Sub training times",
           condition="Type", value="Reward")


# In[60]:

plot.get_figure().savefig(fig_path, format='pdf')


# In[ ]:




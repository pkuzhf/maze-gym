
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

# In[140]:

def plot(log_path):
   procesed_log_path = log_path.replace('.log', '.processed.log')
   fig_path = log_path.replace('.log', '.pdf')
   is_env = True
   test_log_begin = False
   is_map_begin = False
   procesed_log = {'Environment':[], 'Agent':[]}
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
               if line.startswith('(\'env_step\''):
                   is_map_begin = True
                   continue
               if is_map_begin is True:
                   if line.startswith('agent rewards'):
                       continue
                   map_string += line
               if line.startswith('Episode '):
                   is_map_begin = False
                   reward = float(line.split(',')[0].split(': ')[-1])
                   sub_test_times += 1
                   if is_env:
                       procesed_log['Environment'].append({'global_test_times': global_env_test_times, 'sub_test_times': sub_test_times,'reward': reward, 'map_string': map_string}) 
                   if not is_env:
                       procesed_log['Agent'].append({'global_test_times': global_agent_test_times, 'sub_test_times': sub_test_times,'reward': reward, 'map_string': map_string}) 
                   map_string = ''
   with open(procesed_log_path, 'w') as fout:
       json.dump(procesed_log, fout)
   max_test_times = procesed_log['Environment'][-1]['global_test_times']
   total_point = 100
   df = []
   skip_factor = int(max_test_times/total_point)
   if skip_factor < 1:
       skip_factor = 1
   accumulate_rewards = [0.] * 100
   for key in procesed_log.keys():
       logs = procesed_log[key]
       for log in logs:
           accumulate_rewards[log['sub_test_times']] += log['reward']
           if log['global_test_times'] % skip_factor == 0:
               df.append([log['global_test_times'], key, log['sub_test_times'], accumulate_rewards[log['sub_test_times']]/skip_factor])
               accumulate_rewards = [0.] * 100
   df = pd.DataFrame(df)    
   df.columns = ['Training times', 'Type', 'Sub training times', 'Reward']
   # df.count()
   tmp = df[['Training times', 'Sub training times']].groupby(['Training times'])
   minimum_subtimes = tmp['Sub training times'].agg({'max' : np.max})['max'].min()
   sns.plt.rcParams['figure.figsize']=(10,100)
   plt.rcParams['figure.figsize']=(10,1000)
   sns.set(style="whitegrid",font_scale=2)

   plot = sns.tsplot(data=df[df['Sub training times'] <= minimum_subtimes], time="Training times", unit="Sub training times",
              condition="Type", value="Reward")
   plot.get_figure().savefig(fig_path, format='pdf')
   plt.show()
   plot.clear()


# In[141]:

for fd in glob.glob("./logs/*.processed.log"):
    os.remove(fd)
paths = glob.glob('./logs/*.log')
for log_path in paths:
    plot(log_path)


# In[ ]:




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



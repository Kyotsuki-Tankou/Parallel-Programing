import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import csv
import pandas as pd
import numpy as np

matplotlib.rcParams['font.family']='Times New Roman'
data=pd.read_csv('data.csv')
cmap = cm.get_cmap('rainbow')

lines=data['j'].unique()

plt.xlabel('Scale (Square Rooted)')
# plt.xlabel('Thread (Log based on 2)')
plt.suptitle('Sort Efficiency of Different Algorithms in Different Threads (II)',y=0.95)
# plt.suptitle('Sort Efficiency with/without AVX in Different Threads (II)',y=0.95)
# plt.ylabel('Time Elapsed (ms)')
plt.ylabel('Accelerate Ratio')
src=["","poolThread","openMP","pThread","std::sort"]    
baseline=data[data['j']==4]
baseline=baseline.sort_values('i')
for line in lines:
    # if line==2:
    #     continue
    linedata=data[data['j']==line]
    ratio=baseline['k'].values/linedata['k']
    color=cmap(float(line)/len(lines))
    # plt.plot(np.log(linedata['i'])/np.log(2),ratio,label=src[line],color=color)
    # plt.plot(np.log(linedata['i'])/np.log(2),linedata['k'],label=src[line],color=color)
    # plt.plot((linedata['i']),linedata['k'],label=src[line],color=color)
    plt.plot((linedata['i']),ratio,label=src[line],color=color)

plt.legend(loc='lower right')
plt.show()
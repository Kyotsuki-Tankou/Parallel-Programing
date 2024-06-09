import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import csv
import pandas as pd
import numpy as np

matplotlib.rcParams['font.family']='Times New Roman'
data=pd.read_csv('data.csv',sep=' ')
cmap = cm.get_cmap('rainbow')
# data['num']-=5
lines=data['t'].unique()

# plt.xlabel('Scale (Square Rooted)')
# plt.xlabel('Size of the Matrix (Squared and Log based on 2)')
plt.xlabel('Sparse Rate (Log based on 2)')

plt.suptitle('Multiplication Efficiency in Different Operation Systems (II)',y=0.95)
# plt.suptitle('Multiplication Efficiency with/without AVX&openMP in Different Mat-Sizes (I)',y=0.95)

# plt.ylabel('Time Elapsed (ms)')
plt.ylabel('Accelerate Ratio')
# src=["naive","AVX","naive","openMP+AVX","openMP","naive MPI","Bcast MPI","Cyclic MPI"] 
src=["","naive&AVX","naive","openMP&AVX","openMP","MPI-Send&Recv","MPI-Bcast","MPI-Cyclic"] 

baseline=data[data['t']==2]
baseline=baseline.sort_values('time')

for line in lines:
    # if line==6:
    #     continue
    linedata=data[data['t']==line]
    ratio=baseline['time'].values/linedata['time']
    color=cmap(float(line)/(max(lines)))
    plt.plot(np.log(linedata['r0'])/np.log(2),ratio,label=src[line],color=color)
    # plt.plot(np.log(linedata['r0'])/np.log(2),linedata['time'],label=src[line],color=color)
    # plt.plot((linedata['i']),linedata['k'],label=src[line],color=color)
    # plt.plot((linedata['t']),ratio,label=src[line],color=color)

plt.legend(loc='upper left')
plt.show()
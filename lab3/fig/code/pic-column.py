import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.figure(figsize=(8,5))
matplotlib.rcParams['font.family']='Times New Roman'
plt.xlabel('Task-Thread Ratio')
plt.suptitle('Sort Efficiency of Different Algorithms in Different Task-Thread Ratio (II)',y=0.95)
plt.ylabel('Accelerate Ratio')
# plt.ylabel('Time Elapsed (ms)')
data = pd.read_csv('data.csv')
bars = data['j'].unique()
clusters = data['i'].unique()
width = 0.8 / len(bars)

cmap = plt.get_cmap('gray', len(bars)+4)
legend_elements = []
for i, cluster in enumerate(clusters):
    cluster_data = data[data['i'] == cluster]
    for j, bar in enumerate(bars):
        bar_data = cluster_data[cluster_data['j'] == bar]
        if bar_data['k'].values[0] == 0:
            color = 'red'
            height = 8
            label = 'crash'
        else:
            color = cmap(j+2)
            height = 8194.0/bar_data['k']#
            label = f'{bar}'
        plt.bar(i + j * width, height, width=width, color=color, edgecolor='black')
        if (color, label) not in legend_elements:
            legend_elements.append((color, label))

plt.xticks(np.arange(len(clusters)) + width * (len(bars) - 1) / 2, clusters)

plt.legend([plt.Line2D([0], [0], color=color, lw=4) for color, _ in legend_elements],
           [label for _, label in legend_elements], loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

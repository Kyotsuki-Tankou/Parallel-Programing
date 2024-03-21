import pandas as pd
import matplotlib.pyplot as plt

csv_file = 'res.xlsx'
df_prob1 = pd.read_excel(csv_file, sheet_name='prob1')
df_prob2 = pd.read_excel(csv_file, sheet_name='prob2')

# hw_cols = [col for col in df_prob1.columns if col.startswith('hw')]
hw_cols = [col for col in df_prob1.columns if col !='num']

# colors = ['r','g','b']
colors = ['#A2E5EF', '#0BF0E1', 'b', '#F79422','#E5B554','#FFF5AE'] 

plt.figure(figsize=(10, 6))
for i, col in enumerate(hw_cols):
    plt.plot(df_prob1['num'], df_prob1[col], color=colors[i], label=col)
plt.xlabel('num')
plt.ylabel('hw value')
plt.title('prob1')
plt.legend()

plt.figure(figsize=(10, 6))
for i, col in enumerate(hw_cols):
    plt.plot(df_prob2['num'], df_prob2[col], color=colors[i], label=col)
plt.xlabel('num')
plt.ylabel('hw value')
plt.title('prob2')
plt.legend()

plt.show()

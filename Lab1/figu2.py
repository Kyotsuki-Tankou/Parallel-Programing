import pandas as pd
import matplotlib.pyplot as plt

xlsx_file = 'res.xlsx'
df_prob1 = pd.read_excel(xlsx_file, sheet_name='prob1')
df_prob2 = pd.read_excel(xlsx_file, sheet_name='prob2')

# hw_cols = [col for col in df_prob1.columns if col.startswith('hw')]
hw_cols = [col for col in df_prob1.columns if col !='num']
# colors = ['r','g','b']
colors = ['#A2E5EF', '#0BF0E1', 'b', '#F79422','#E5B554','#FFF5AE'] 

fig, axs = plt.subplots(1, 3, figsize=(18, 5)) 
for i, (rng_start, rng_end, label) in enumerate([(10, 100, '10-100'), (100, 1000, '100-1000'), (1000, 10000, '1000-10000')]):
    df = df_prob1.loc[(df_prob1['num'] >= rng_start) & (df_prob1['num'] < rng_end), :]
    for j, col in enumerate(hw_cols):
        axs[i].plot(df['num'], df[col], color=colors[j], label=col)
    axs[i].set_xlabel('num')
    # axs[i].set_ylabel('hw value')
    axs[i].set_ylabel('all value')
    axs[i].set_title(f'prob1 ({label})')
    axs[i].legend()

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for i, (rng_start, rng_end, label) in enumerate([(10, 16, '10-16'), (17, 22, '17-22'), (23,29,'23-29')]):
    df = df_prob2.loc[(df_prob2['num'] >= rng_start) & (df_prob2['num'] < rng_end), :]
    for j, col in enumerate(hw_cols):
        axs[i].plot(df['num'], df[col], color=colors[j], label=col)
    axs[i].set_xlabel('num')
    # axs[i].set_ylabel('hw value')
    axs[i].set_ylabel('all value')
    axs[i].set_title(f'prob2 ({label})')
    axs[i].legend()

plt.show()

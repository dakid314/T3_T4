'''
Author: George Zhao
Date: 2022-11-03 21:28:14
LastEditors: George Zhao
LastEditTime: 2022-11-06 15:34:32
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# %%
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
# %%
df = pd.read_csv('out/rocauc1.csv', index_col=0).T
df.columns = ['r' + i for i in df.columns]
# %%
plt.figure(figsize=(21.6 * 1.2, 2.7 * 4))
sns.heatmap(df, vmin=0.6, vmax=1, cmap='seismic',
            linewidths=.5,
            fmt='',
            annot=df.applymap(
                lambda x: f'{round(x, 2):3.2f}'.replace('0.', '.')),
            square=False
            )
plt.xticks(fontsize=16)
plt.yticks(fontsize=16, rotation=0)
plt.tight_layout()
plt.savefig('out/rocauc2.svg', transparent=True)
# %%

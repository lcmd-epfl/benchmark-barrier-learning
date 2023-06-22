import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
# read results
cyclo_true_df = pd.read_csv('results/cyclo_true/test_scores.csv')
cyclo_rxnmapper_df = pd.read_csv('results/cyclo_all/test_scores.csv')
cyclo_random_df = pd.read_csv('results/cyclo_random/test_scores.csv')
cyclo_spahm = np.load('data/cyclo/spahm_10_fold.npy')

# missing gdb true but around 3 kcal
gdb_true_df = pd.read_csv('results/gdb_true/test_scores.csv')
gdb_rxnmapper_df = pd.read_csv('results/gdb_all/test_scores.csv')
gdb_random_df = pd.read_csv('results/gdb_random/test_scores.csv')
gdb_spahm = np.load('data/gdb7-22-ts/spahm_10_fold.npy')

proparg_random_df = pd.read_csv('results/proparg/test_scores.csv')
proparg_spahm = np.load('data/proparg/spahm_10_fold.npy')

#matplotlib.rcParams["figure.figsize"] = (10, 4.4)
matplotlib.rcParams.update({"font.size":11})
colors = ["#FF0000", "#B51F1F", "#00A79F", "#007480", "#413D3A", "#CAC7C7"]

fig, axes = plt.subplots(nrows=1, ncols=3)

ax = axes[1]
ax.set_title('(b) Cyclo-23-TS')
labels = ['CGR True', 'CGR RXNMapper', 'CGR Random', 'SPA$^H$M$_b$+KRR']
for i, df in enumerate([cyclo_true_df, cyclo_rxnmapper_df, cyclo_random_df]):
    ax.bar(i, df['Mean mae'], yerr=df['Standard deviation mae'], color=colors[i])
ax.bar(3, np.mean(cyclo_spahm), yerr=np.std(cyclo_spahm), color=colors[4])
ax.set_xticks(list(range(len(labels))))
ax.set_xticklabels(labels, rotation=90)

ax = axes[0]
ax.set_title('(a) GDB7-22-TS')
ax.set_ylabel("MAE $\Delta E^\ddag$ [kcal/mol]")
for i, df in enumerate([gdb_true_df, gdb_rxnmapper_df, gdb_random_df]):
    ax.bar(i, df['Mean mae'], yerr=df['Standard deviation mae'], color=colors[i])
ax.bar(3, np.mean(gdb_spahm), yerr=np.std(gdb_spahm), color=colors[4])
ax.set_xticks(list(range(len(labels))))
ax.set_xticklabels(labels, rotation=90)

ax = axes[2]
ax.set_title('(c) Proparg-21-TS')
for i in range(2):
    ax.bar(i, 0, color=colors[i])
ax.bar(2, proparg_random_df['Mean mae'], yerr=proparg_random_df['Standard deviation mae'], color=colors[2])
ax.bar(3, np.mean(proparg_spahm), yerr=np.std(proparg_spahm), color=colors[4])
ax.set_xticks([0,1,2,3])
ax.set_xticklabels(['', '', 'CGR Random', 'SPA$^H$M$_b$+KRR'], rotation=90)
figname = 'figures/atom_mapping_quality.pdf'

plt.tight_layout()
plt.savefig(figname)
plt.show()
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

# read results
cyclo_true_df = pd.read_csv('results/cyclo_true/test_scores.csv')
cyclo_rxnmapper_df = pd.read_csv('results/cyclo_rxnmapper/test_scores.csv')
cyclo_random_df = pd.read_csv('results/cyclo_random/test_scores.csv')
cyclo_slatm = np.load('data/cyclo/slatm_10_fold.npy')

gdb_true_df = pd.read_csv('results/gdb_true/test_scores.csv')
gdb_rxnmapper_df = pd.read_csv('results/gdb_rxnmapper/test_scores.csv')
gdb_random_df = pd.read_csv('results/gdb_random/test_scores.csv')
gdb_slatm = np.load('data/gdb7-22-ts/slatm_10_fold.npy')

proparg_true_df = pd.read_csv('results/proparg_true/test_scores.csv')
proparg_random_df = pd.read_csv('results/proparg_random/test_scores.csv')
proparg_slatm = np.load('data/proparg/slatm_10_fold.npy')

#matplotlib.rcParams["figure.figsize"] = (10, 4.4)
matplotlib.rcParams.update({"font.size":13})
colors = ["#FF0000", "#B51F1F", "#00A79F", "#007480", "#413D3A", "#CAC7C7"]

fig, axes = plt.subplots(nrows=1, ncols=3)

ax = axes[1]
ax.set_ylabel("MAE $\Delta G^\ddag$ [kcal/mol]")

ax.set_title('(b) Cyclo-23-TS', fontsize='medium')

labels = ['CGR True', 'CGR RXNMapper', 'CGR Random', 'SLATM$_d$']

colors = ['blue', 'purple', 'magenta', "#00A79F"]
ax.set_ylabel("MAE $\Delta G^\ddag$ [kcal/mol]")

for i, df in enumerate([cyclo_true_df, cyclo_rxnmapper_df, cyclo_random_df]):
    ax.bar(i, df['Mean mae'], yerr=df['Standard deviation mae'], color=colors[i])
ax.bar(3, np.mean(cyclo_slatm), yerr=np.std(cyclo_slatm), color=colors[3])
ax.set_xticks(list(range(len(labels))))
ax.set_xticklabels(labels, rotation=90)

ax = axes[0]
ax.set_title('(a) GDB7-22-TS', fontsize='medium')
ax.set_ylabel("MAE $\Delta E^\ddag$ [kcal/mol]")
for i, df in enumerate([gdb_true_df, gdb_rxnmapper_df, gdb_random_df]):
    ax.bar(i, df['Mean mae'], yerr=df['Standard deviation mae'], color=colors[i])
ax.bar(3, np.mean(gdb_slatm), yerr=np.std(gdb_slatm), color=colors[3])
ax.set_xticks(list(range(len(labels))))
ax.set_xticklabels(labels, rotation=90)

ax = axes[2]

ax.set_title('(c) Proparg-21-TS', fontsize='medium')
ax.set_ylabel("MAE $\Delta E^\ddag$ [kcal/mol]")

ax.bar(0, proparg_true_df['Mean mae'], yerr=proparg_true_df['Standard deviation mae'], color=colors[0])
ax.bar(1, 0, color=colors[1])
ax.bar(2, proparg_random_df['Mean mae'], yerr=proparg_random_df['Standard deviation mae'], color=colors[2])
ax.bar(3, np.mean(proparg_slatm), yerr=np.std(proparg_slatm), color=colors[3])
ax.set_xticks([0,1,2,3])
ax.set_xticklabels(['CGR True', '', 'CGR Random', 'SLATM$_d$'], rotation=90)
figname = 'figures/atom_mapping_quality.pdf'

plt.tight_layout()
plt.savefig(figname)
#plt.show()

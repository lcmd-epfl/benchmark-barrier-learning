import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

proparg = False

# read results
cyclo_true_df = pd.read_csv('results/cyclo_true/test_scores.csv')
cyclo_rxnmapper_df = pd.read_csv('results/cyclo_all/test_scores.csv')
cyclo_random_df = pd.read_csv('results/cyclo_random/test_scores.csv')

# missing gdb true but around 3 kcal
gdb_true_df = pd.read_csv('results/gdb_true/test_scores.csv')
gdb_rxnmapper_df = pd.read_csv('results/gdb_all/test_scores.csv')
gdb_random_df = pd.read_csv('results/gdb_random/test_scores.csv')

proparg_random_df = pd.read_csv('results/proparg/test_scores.csv')

#matplotlib.rcParams["figure.figsize"] = (10, 4.4)
matplotlib.rcParams.update({"font.size":14})
colors = ["#FF0000", "#B51F1F", "#00A79F", "#007480", "#413D3A", "#CAC7C7"]

if proparg:
    fig, axes = plt.subplots(nrows=1, ncols=3)
else:
    fig, axes = plt.subplots(nrows=1, ncols=2)

ax = axes[0]
ax.set_ylabel("MAE $\Delta G^\ddag$ [kcal/mol]")
ax.set_title('Cyclo-23-TS')
labels = ['True', 'RXNMapper', 'Random']
for i, df in enumerate([cyclo_true_df, cyclo_rxnmapper_df, cyclo_random_df]):
    ax.bar(i, df['Mean mae'], yerr=df['Standard deviation mae'], color=colors[i])
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=45)

ax = axes[1]
ax.set_title('GDB7-22-TS')
for i, df in enumerate([gdb_true_df, gdb_rxnmapper_df, gdb_random_df]):
    ax.bar(i, df['Mean mae'], yerr=df['Standard deviation mae'], color=colors[i])
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=45)

figname = 'figures/atom_mapping_quality.pdf'

if proparg:
    ax = axes[2]
    ax.set_title('Proparg-21-TS')
    for i in range(2):
        ax.bar(i, 0, color=colors[i])
    ax.bar(2, proparg_random_df['Mean mae'], yerr=proparg_random_df['Standard deviation mae'], color=colors[2])
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['', '', 'Random'])
    figname = 'figures/atom_mapping_quality_w_proparg.pdf'

plt.tight_layout()
plt.savefig(figname)
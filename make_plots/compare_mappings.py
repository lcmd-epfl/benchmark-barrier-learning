import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

datasets = ['gdb', 'cyclo', 'proparg']
labels = ['CGR True', 'CGR RXNMapper', 'CGR Random', 'SLATM$_d$']
keys = ['true_df', 'rxnmapper_df', 'random_df', 'slatm']
#colors = ["#FF0000", "#B51F1F", "#00A79F", "#007480", "#413D3A", "#CAC7C7"]
colors = ['blue', 'purple', 'magenta', "#00A79F"]

data = {}
for dataset in datasets:
    data[dataset] = {}
    for key in keys[:-1]:
        data[dataset][key] = pd.read_csv(f'results/{dataset}_{key[:-3]}/test_scores.csv')
    slatm = np.load(f'data/{"gdb7-22-ts" if dataset=="gdb" else dataset}/slatm_10_fold.npy')
    data[dataset]['slatm'] = {'Mean mae': np.mean(slatm), 'Standard deviation mae': np.std(slatm)}

data['gdb']['title'] = '(a) GDB7-22-TS'
data['cyclo']['title'] = '(b) Cyclo-23-TS'
data['proparg']['title'] = '(c) Proparg-21-TS'

data['gdb']['ylabel'] = "MAE $\Delta E^\ddag$ [kcal/mol]"
data['cyclo']['ylabel'] = "MAE $\Delta G^\ddag$ [kcal/mol]"
data['proparg']['ylabel'] = "MAE $\Delta E^\ddag$ [kcal/mol]"

matplotlib.rcParams.update({"font.size":13})
fig, axes = plt.subplots(nrows=1, ncols=3)

for ax, dataset_name in zip(axes, datasets):

    dataset = data[dataset_name]
    ax.set_title(dataset['title'], fontsize='medium')
    ax.set_ylabel(dataset['ylabel'])

    for i, key in enumerate(keys):
        df = dataset[key]
        ax.bar(i, df['Mean mae'], yerr=df['Standard deviation mae'], color=colors[i])
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=90)

figname = 'figures/atom_mapping_quality.pdf'
plt.tight_layout()
#plt.savefig(figname)
#plt.show()

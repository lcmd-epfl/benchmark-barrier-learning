import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

def round_with_std(mae, std):
    # works only is std < 1
    std_1digit = int(std // 10**np.floor(np.log10(std)))
    std_1digit_pos = f'{std:.10f}'.split('.')[1].index(str(std_1digit))
    if std_1digit > 2:
        std_round = f'{std:.{std_1digit_pos+1}f}'
    else:
        std_round = f'{std:.{std_1digit_pos+2}f}'
    n_after_point = len(std_round.split('.')[1])
    mae_round = f'{mae:.{n_after_point}f}'
    return mae_round + '$\pm$' + std_round

datasets = ['gdb', 'cyclo', 'proparg']
labels = ['Chemprop True', 'Chemprop RXNMapper', 'Chemprop None']
keys = ['true', 'rxnmapper', 'nomap']
#colors = ["#FF0000", "#B51F1F", "#00A79F", "#007480", "#413D3A", "#CAC7C7"]
colors = ['blue', 'purple', 'magenta']

data = {}
for dataset in datasets:
    data[dataset] = {}

    for key in keys:
        data[dataset][key] = {}
        data[dataset][key]['mae'] = 0
        data[dataset][key]['std'] = 0
        if dataset == 'cyclo' or (dataset == 'gdb' and key != 'true') or key == 'rxnmapper' or (dataset == 'proparg' and key == 'nomap'):
            info = pd.read_csv(f'results/{dataset}_{key}/test_scores.csv')
        else:
            info = pd.read_csv(f'results/{dataset}_{key}_withH/test_scores.csv')
        data[dataset][key]['mae'] = info['Mean mae']
        data[dataset][key]['std'] = info['Standard deviation mae']

data['gdb']['title'] = '(a) GDB7-22-TS'
data['cyclo']['title'] = '(b) Cyclo-23-TS'
data['proparg']['title'] = '(c) Proparg-21-TS'

data['gdb']['ylabel'] = "MAE $\Delta E^\ddag$ [kcal/mol]"
data['cyclo']['ylabel'] = "MAE $\Delta G^\ddag$ [kcal/mol]"
data['proparg']['ylabel'] = "MAE $\Delta E^\ddag$ [kcal/mol]"

matplotlib.rcParams.update({"font.size":12})
fig, axes = plt.subplots(nrows=1, ncols=3)

axes[0].set_ylim(0,13)
axes[1].set_ylim(0,4.2)
axes[2].set_ylim(0,2.5)
adds = [0.39, 0.22, 0.13]
for ax, dataset_name, add in zip(axes, datasets, adds):

    dataset = data[dataset_name]

    ax.set_title(dataset['title'], fontsize='medium')
    ax.set_ylabel(dataset['ylabel'])

    for i, key in enumerate(keys):
        df = dataset[key]
        ax.bar(i, df['mae'], yerr=df['std'], color=colors[i])
        ax.text(i - 0.11, df['mae'] + add, round_with_std(float(df['mae']), float(df['std'])), fontsize='x-small', fontweight='bold', rotation=90)

    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=90, fontsize=10)

figname = 'figures/atom_mapping_quality.pdf'
plt.tight_layout()
plt.savefig(figname)
plt.show()

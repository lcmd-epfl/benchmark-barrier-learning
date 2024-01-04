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
labels = ['Chemprop True', 'Chemprop RXNMapper', 'Chemprop None', 'SLATM$_d$+KRR']
keys = ['true_df', 'rxnmapper_df', 'nomap_df', 'slatm']
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

matplotlib.rcParams.update({"font.size":12})
fig, axes = plt.subplots(nrows=1, ncols=3)

add = 0.17
axes[0].set_ylim(0,12.5)
axes[1].set_ylim(0,4.2)
axes[2].set_ylim(0,2.5)
for ax, dataset_name in zip(axes, datasets):

    dataset = data[dataset_name]
    ax.set_title(dataset['title'], fontsize='medium')
    ax.set_ylabel(dataset['ylabel'])

    for i, key in enumerate(keys):
        df = dataset[key]
        ax.bar(i, df['Mean mae'], yerr=df['Standard deviation mae'], color=colors[i])
        ax.text(i - 0.2, df['Mean mae'] + add, round_with_std(float(df['Mean mae']), float(df['Standard deviation mae'])), fontsize='x-small', fontweight='bold', rotation=90)

    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=90, fontsize=10)

figname = 'figures/atom_mapping_quality.pdf'
plt.tight_layout()
plt.savefig(figname)
plt.show()

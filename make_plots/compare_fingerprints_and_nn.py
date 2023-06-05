import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

def get_maes(npy, txt=False, csv=False):
    if txt:
        maes = np.loadtxt(npy)
    elif csv:
        df = pd.read_csv(npy)
        return df['Mean mae'], df['Standard deviation mae']
    else:
        maes = np.load(npy)
    return np.mean(maes), np.std(maes)

matplotlib.rcParams.update({"font.size":14})
colors = ["#FF0000", "#B51F1F", "#00A79F", "#007480", "#413D3A", "#CAC7C7"]

cyclo_dir = 'data/cyclo/'
gdb_dir = 'data/gdb7-22-ts/'

cyclo_lang_dir = 'outs/cyclo_bert_pretrained/results.txt'
gdb_lang_dir = 'outs/gdb_bert_pretrained/results.txt'
lang_dirs = [cyclo_lang_dir, gdb_lang_dir]

cyclo_cgr_dir = 'results/cyclo_true/test_scores.csv'
gdb_cgr_dir = 'results/gdb_true/test_scores.csv'
cgr_dirs = [cyclo_cgr_dir, gdb_cgr_dir]

titles = ['Cyclo-23-TS', 'GDB7-22-TS']
fig, axes = plt.subplots(nrows=1, ncols=2)
for i, db in enumerate([cyclo_dir, gdb_dir]):
    lang_dir = lang_dirs[i]
    cgr_dir = cgr_dirs[i]
    mfp_mae, mfp_std = get_maes(db + 'mfp_10_fold.npy')
    drfp_mae, drfp_std = get_maes(db + 'drfp_10_fold.npy')
    slatm_mae, slatm_std = get_maes(db + 'slatm_10_fold.npy')
    b2r2_mae, b2r2_std = get_maes(db + 'b2r2_10_fold.npy')
    spahm_mae, spahm_std = get_maes(db + 'spahm_10_fold.npy')

    rxnfp_mae, rxnfp_std = get_maes(lang_dir, txt=True)
    cgr_mae, cgr_std = get_maes(cgr_dir, csv=True)

    axes[i].set_title(titles[i])
    axes[i].bar(0, mfp_mae, yerr=mfp_std, color=colors[0])
    axes[i].bar(1, drfp_mae, yerr=drfp_std, color=colors[1])
    axes[i].bar(2, slatm_mae, yerr=slatm_std, color=colors[2])
    axes[i].bar(3, b2r2_mae, yerr=b2r2_std, color=colors[3])
    axes[i].bar(4, spahm_mae, yerr=spahm_std, color=colors[4])

    axes[i].bar(5, rxnfp_mae, yerr=rxnfp_std, color=colors[5])
    axes[i].bar(6, cgr_mae, yerr=cgr_std, color='blue')

    axes[i].set_xticks(list(range(7)))
    axes[i].set_xticklabels(['MFP', 'DRFP', 'SLATM', '$B^2R^2_l$', 'SPA$^H$M$_b$', 'RXNFP', 'CGR'], rotation=90)
axes[0].set_ylabel("MAE $\Delta G^\ddag$ [kcal/mol]")

plt.tight_layout()
plt.savefig('figures/compare_all_models.pdf')
plt.show()


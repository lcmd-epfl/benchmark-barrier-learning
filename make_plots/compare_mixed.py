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

matplotlib.rcParams.update({"font.size":11})
colors = ["#FF0000", "#B51F1F", "#00A79F", "#007480", "#413D3A", "#CAC7C7"]

cyclo_dir = 'data/cyclo/'
gdb_dir = 'data/gdb7-22-ts/'
proparg_dir = 'data/proparg/'

titles = ['(a) GDB7-22-TS', '(b) Cyclo-23-TS', '(c) Proparg-21-TS']
fig, axes = plt.subplots(nrows=1, ncols=3)
for i, db in enumerate([gdb_dir, cyclo_dir, proparg_dir]):

    mfp_mae, mfp_std = get_maes(db + 'mfp_10_fold.npy')
    spahm_mae, spahm_std = get_maes(db + 'spahm_10_fold.npy')
    mixed_mae, mixed_std = get_maes(db + 'mixed_10_fold.npy')

    axes[i].set_title(titles[i])
    axes[i].bar(0, mfp_mae, yerr=mfp_std, color=colors[0])
    axes[i].bar(1, spahm_mae, yerr=spahm_std, color=colors[4])
    axes[i].bar(2, mixed_mae, yerr=mixed_std, color='purple')

    axes[i].set_xticks(list(range(3)))
    axes[i].set_xticklabels(['MFP', 'SPA$^H$M$_b$', 'MFP+SPA$^H$M$_b$'], rotation=90)
axes[0].set_ylabel("MAE $\Delta E^\ddag$ [kcal/mol]")

plt.tight_layout()
plt.savefig('figures/compare_mixed.pdf')
plt.show()


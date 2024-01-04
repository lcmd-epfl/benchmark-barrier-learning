import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

def get_maes(npy, txt=False, csv=False):
    if txt:
        maes = np.loadtxt(npy)
    elif csv:
        df = pd.read_csv(npy)
        return df['Mean mae'].to_numpy()[0], df['Standard deviation mae'].to_numpy()[0]
    else:
        maes = np.load(npy)
    return np.mean(maes), np.std(maes)

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

matplotlib.rcParams.update({"font.size":12})
colors = ["#FF0000", "#B51F1F", "#00A79F", "#007480", "#413D3A", "#CAC7C7"]


cyclo_dir = 'data/cyclo/'
gdb_dir = 'data/gdb7-22-ts/'
proparg_dir = 'data/proparg/'

cyclo_std = np.std(pd.read_csv(cyclo_dir+'full_dataset.csv')['G_act'].to_numpy())
gdb_std = np.std(pd.read_csv("data/gdb7-22-ts/ccsdtf12_dz.csv")['dE0'].to_numpy())
proparg_std = np.std(pd.read_csv("data/proparg/data.csv")['Eafw'].to_numpy())
stds = [gdb_std, cyclo_std, proparg_std]

titles = ['(a) GDB7-22-TS', '(b) Cyclo-23-TS', '(c) Proparg-21-TS']
fig, axes = plt.subplots(nrows=3, ncols=1)

axes[0].set_xlim(0,13)
axes[1].set_xlim(0,6)
axes[2].set_xlim(0,1.4)
for i, db in enumerate([gdb_dir, cyclo_dir, proparg_dir]):

    if i == 0:
        add = 0.23
    elif i == 1:
        add = 0.15
    elif i == 2:
        add = 0.1

    slatm_mae_xtb, slatm_std_xtb = get_maes(db + 'slatm_10_fold_xtb.npy')
    slatm_mae, slatm_std = get_maes(db + 'slatm_10_fold.npy')

    b2r2_mae_xtb, b2r2_std_xtb = get_maes(db + 'b2r2_l_10_fold_xtb.npy')
    b2r2_mae, b2r2_std = get_maes(db + 'b2r2_l_10_fold.npy')

    axes[i].set_title(titles[i], fontsize='medium')
    #axes[i].axhline(stds[i], color='black', alpha=0.5, linestyle='dashed', label='std')

    axes[i].barh(3, slatm_mae, xerr=slatm_std, color=colors[2])
    axes[i].text(slatm_mae + add, 2.8, round_with_std(slatm_mae, slatm_std), fontsize='x-small', fontweight='bold')

    axes[i].barh(2, slatm_mae_xtb, xerr=slatm_std_xtb, color=colors[2], hatch='/')
    axes[i].text(slatm_mae_xtb + add, 1.8, round_with_std(slatm_mae_xtb, slatm_std_xtb), fontsize='x-small', fontweight='bold')

    axes[i].barh(1, b2r2_mae, xerr=b2r2_std, color=colors[3])
    axes[i].text(b2r2_mae + add, 0.8, round_with_std(b2r2_mae, b2r2_std), fontsize='x-small', fontweight='bold')

    axes[i].barh(0, b2r2_mae_xtb, xerr=b2r2_std_xtb, color=colors[3], hatch='/')
    axes[i].text(b2r2_mae_xtb + add, -0.2, round_with_std(b2r2_mae_xtb, b2r2_std_xtb), fontsize='x-small', fontweight='bold')

    axes[i].set_yticks(list(range(4)))
    axes[i].set_yticklabels(['SLATM$_d$+KRR(dft)', 'SLATM$_d$+KRR(xtb)', '$B^2R^2_l$+KRR(dft)', '$B^2R^2_l$+KRR(xtb)'], fontsize=10)

axes[0].set_xlabel("MAE $\Delta E^\ddag$ [kcal/mol]", fontsize=12)
axes[1].set_xlabel("MAE $\Delta G^\ddag$ [kcal/mol]", fontsize=12)
axes[2].set_xlabel("MAE $\Delta E^\ddag$ [kcal/mol]", fontsize=12)

plt.tight_layout()
plt.savefig('figures/compare_all_models_xtb.pdf')
plt.show()

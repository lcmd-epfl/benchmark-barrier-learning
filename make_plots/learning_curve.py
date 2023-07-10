import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# load LCs
tr_sizes = np.load('data/gdb7-22-ts/tr_sizes.npy')
slatm_lc_rbf = np.load('data/gdb7-22-ts/slatm_lc_10_fold.npy')
spahm_a_lc = np.load('data/gdb7-22-ts/spahm_a_lc_10_fold.npy')
spahm_b_lc = np.load('data/gdb7-22-ts/spahm_b_lc_10_fold.npy')
spahm_e_lc = np.load('data/gdb7-22-ts/spahm_e_lc_10_fold.npy')

matplotlib.rcParams.update({"font.size":14})
colors = ["#FF0000", "#B51F1F", "#00A79F", "#007480", "#413D3A", "#CAC7C7"]

fig,ax = plt.subplots(nrows=1, ncols=1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.errorbar(tr_sizes, np.mean(slatm_lc_rbf,axis=0), yerr=np.std(slatm_lc_rbf, axis=0), color=colors[0], label='SLATM', marker='.', markersize=10)
ax.errorbar(tr_sizes, np.mean(spahm_a_lc, axis=0), yerr=np.std(spahm_a_lc, axis=0), color=colors[4], label='SPA$^H$M$_a$', marker='.', markersize=10)
ax.errorbar(tr_sizes, np.mean(spahm_b_lc, axis=0), yerr=np.std(spahm_b_lc, axis=0),color=colors[2], label='SPA$^H$M$_b$', marker='.', markersize=10)
ax.errorbar(tr_sizes, np.mean(spahm_e_lc, axis=0), yerr=np.std(spahm_e_lc, axis=0), color=colors[3], label='SPA$^H$M$_e$', marker='.', markersize=10)

xticks = [900, 3000,10000]
ax.set_xticks(xticks)
ax.set_xticklabels([str(x) for x in xticks])

yticks = [7,12,20]
ax.set_yticks(yticks)
ax.set_yticklabels([str(x) for x in yticks])

ax.minorticks_off()
ax.set_ylabel("MAE $\Delta E^\ddag$ [kcal/mol")
ax.set_xlabel("$N_{\mathrm{train}}$")
ax.legend()

plt.tight_layout()
plt.savefig('figures/learning_curves_gdb.pdf')
plt.show()
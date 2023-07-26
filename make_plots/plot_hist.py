import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

gdb_df = pd.read_csv("data/gdb7-22-ts/ccsdtf12_dz.csv")
gdb_y = gdb_df['dE0'].to_numpy()

cyclo_df = pd.read_csv("data/cyclo/full_dataset.csv")
cyclo_y = cyclo_df['G_act'].to_numpy()

proparg_df = pd.read_csv("data/proparg/data.csv")
proparg_y = proparg_df['Eafw'].to_numpy()

proparg_nbins = 10
dy = (max(proparg_y)-min(proparg_y))/proparg_nbins
cyclo_nbins = int(round((max(cyclo_y)-min(cyclo_y))/dy))
gdb_nbins   = int(round((max(gdb_y)-min(gdb_y))/dy))

matplotlib.rcParams.update({"font.size":14})
colors = ["#FF0000", "#B51F1F", "#00A79F", "#007480", "#413D3A", "#CAC7C7"]

fig, ax = plt.subplots(nrows=1, ncols=1)

ax.hist(gdb_y, label='GDB7-22-TS', color=colors[1], bins=gdb_nbins, edgecolor='black', histtype='stepfilled', linewidth=0.75)
ax.hist(cyclo_y, label='Cyclo-23-TS', color=colors[3], bins=cyclo_nbins, edgecolor='black', histtype='stepfilled', linewidth=0.75)
ax.hist(proparg_y, label='Proparg-21-TS', color=colors[5], bins=proparg_nbins, edgecolor='black', histtype='stepfilled', linewidth=0.75)
ax.set_ylabel("Count")
ax.set_xlabel("$\Delta G^\ddag / \Delta E^\ddag$ [kcal/mol]")
plt.legend()
plt.tight_layout()
plt.savefig('figures/hist.pdf')

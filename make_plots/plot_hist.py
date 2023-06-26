import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

gdb_df = pd.read_csv("data/gdb7-22-ts/ccsdtf12_dz.csv")
gdb_y = gdb_df['dE0'].to_numpy()

cyclo_df = pd.read_csv("data/cyclo/full_dataset.csv")
cyclo_y = cyclo_df['G_act'].to_numpy()

proparg_df = pd.read_csv("data/proparg/data.csv")
proparg_y = proparg_df['Eafw'].to_numpy()

matplotlib.rcParams.update({"font.size":14})
colors = ["#FF0000", "#B51F1F", "#00A79F", "#007480", "#413D3A", "#CAC7C7"]

fig, ax = plt.subplots(nrows=1, ncols=1)

ax.hist(gdb_y, label='GDB7-22-TS', color=colors[1])
ax.hist(cyclo_y, label='Cyclo-23-TS', color=colors[3])
ax.hist(proparg_y, label='Proparg-21-TS', color=colors[5])
ax.set_ylabel("Count")
ax.set_xlabel("$\Delta E^\ddag$ [kcal/mol]")
plt.legend()
plt.tight_layout()
plt.savefig('figures/hist.pdf')
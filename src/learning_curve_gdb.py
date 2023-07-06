from src import learning
from src.reaction_reps import QML, SPAHM
import os
import numpy as np

# 3d fingerprints SLATM and SPAHMb
qml = QML()
qml.get_GDB7_ccsd_data()
barriers = qml.barriers
slatm_save = 'data/gdb7-22-ts/slatm.npy'
if not os.path.exists(slatm_save):
    slatm = qml.get_SLATM()
    np.save(slatm_save, slatm)
else:
    slatm = np.load(slatm_save)

# for now have to load spahm
sp = SPAHM()
sp.get_gdb_data_and_rep()
spahm_b = sp.spahm_b
spahm_a = sp.spahm_a
spahm_e = sp.spahm_e
sp_barriers = sp.barriers

print("reps generated/loaded, predicting")

CV=10
slatm_fname = f'data/gdb7-22-ts/slatm_lc_{CV}_fold.npy'
if not os.path.exists(slatm_fname):
    print("Running SLATM LC...")
    tr_sizes, lc_slatm = learning.learning_curve_KRR(slatm, barriers, CV=CV, save_hypers=True, save_file='data/gdb7-22-ts/slatm_lc_hypers.csv')
    np.save(slatm_fname, lc_slatm)
    np.save('data/gdb7-22-ts/tr_sizes.npy', tr_sizes)


spahm_a_fname = f'data/gdb7-22-ts/spahm_a_lc_{CV}_fold.npy'
if not os.path.exists(spahm_a_fname):
    print("Running SPAHM(a) LC...")
    _, lc_spahm_a = learning.learning_curve_KRR(spahm_a, sp_barriers, CV=CV, save_hypers=True, save_file='data/gdb7-22-ts/spahm_a_lc_hypers.csv')
    np.save(spahm_a_fname, lc_spahm_a)

spahm_b_fname = f'data/gdb7-22-ts/spahm_b_lc_{CV}_fold.npy'
if not os.path.exists(spahm_b_fname):
    print("Running SPAHM(b) LC...")
    _, lc_spahm_b = learning.learning_curve_KRR(spahm_b, sp_barriers, CV=CV, save_hypers=True, save_file='data/gdb7-22-ts/spahm_b_lc_hypers.csv')
    np.save(spahm_b_fname, lc_spahm_b)

spahm_e_fname = f'data/gdb7-22-ts/spahm_e_lc_{CV}_fold.npy'
if not os.path.exists(spahm_e_fname):
    print("Running SPAHM(e) LC...")
    _, lc_spahm_e = learning.learning_curve_KRR(spahm_e, sp_barriers, CV=CV, save_hypers=True, save_file='data/gdb7-22-ts/spahm_e_lc_hypers.csv')
    np.save(spahm_e_fname, lc_spahm_e)
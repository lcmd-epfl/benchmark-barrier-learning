import argparse as ap
from src.reaction_reps import B2R2
from src.learning import predict_CV
import numpy as np
import os

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('-CV', '--CV', default=3)
    parser.add_argument('-tr', '--train', default=0.8)
    args = parser.parse_args()
    args.CV = int(args.CV)
    args.train = float(args.train)
    # maybe hydroform ?
    return args

if __name__ == "__main__":
    args = parse_args()
    CV = args.CV

    for dataset in ['cyclo', 'gdb7-22-ts', 'proparg']:
        print(f"Running for {dataset} dataset")
        b2r2 = B2R2()
        if dataset == 'cyclo':
            b2r2.get_cyclo_data()
        elif dataset == 'gdb7-22-ts':
            b2r2.get_GDB7_ccsd_data()
        elif dataset == 'proparg':
            b2r2.get_proparg_data()
        barriers = b2r2.barriers
        b2r2_save = f'data/{dataset}/b2r2_cut3.npy'
        if not os.path.exists(b2r2_save):
            b2r2_3 = b2r2.get_b2r2_l(Rcut=3)
            np.save(b2r2_save, b2r2_3)
        else:
            b2r2_3 = np.load(b2r2_save)

        b2r2_save = f'data/{dataset}/b2r2_cut4.npy'
        if not os.path.exists(b2r2_save):
            b2r2_4 = b2r2.get_b2r2_l(Rcut=4)
            np.save(b2r2_save, b2r2_4)
        else:
            b2r2_4 = np.load(b2r2_save)

        b2r2_save = f'data/{dataset}/b2r2_cut5.npy'
        if not os.path.exists(b2r2_save):
            b2r2_5 = b2r2.get_b2r2_l(Rcut=5)
            np.save(b2r2_save, b2r2_5)
        else:
            b2r2 = np.load(b2r2_save)

        b2r2_save = f'data/{dataset}/b2r2_cut6.npy'
        if not os.path.exists(b2r2_save):
            b2r2_6 = b2r2.get_b2r2_l(Rcut=6)
            np.save(b2r2_save, b2r2)
        else:
            b2r2_6 = np.load(b2r2_save)

        print("reps generated/loaded, predicting")

        b2r2_save = f'data/{dataset}/b2r2_3_pred.npy'
        if not os.path.exists(b2r2_save):
            maes_b2r2_3 = predict_CV(b2r2_3, barriers, CV=CV, mode='krr', save_hypers=False)
            np.save(b2r2_save, maes_b2r2_3)
        else:
            maes_b2r2_3 = np.load(b2r2_save)
        print(f'b2r2_3 mae {np.mean(maes_b2r2_3)} +- {np.std(maes_b2r2_3)}')

        b2r2_save = f'data/{dataset}/b2r2_4_pred.npy'
        if not os.path.exists(b2r2_save):
            maes_b2r2_4 = predict_CV(b2r2_4, barriers, CV=CV, mode='krr', save_hypers=False)
            np.save(b2r2_save, maes_b2r2_4)
        else:
            maes_b2r2_4 = np.load(b2r2_save)
        print(f'b2r2_4 mae {np.mean(maes_b2r2_4)} +- {np.std(maes_b2r2_4)}')

        b2r2_save = f'data/{dataset}/b2r2_5_pred.npy'
        if not os.path.exists(b2r2_save):
            maes_b2r2_5 = predict_CV(b2r2_5, barriers, CV=CV, mode='krr', save_hypers=False)
            np.save(b2r2_save, maes_b2r2_5)
        else:
            maes_b2r2_5 = np.load(b2r2_save)
        print(f'b2r2_5 mae {np.mean(maes_b2r2_5)} +- {np.std(maes_b2r2_5)}')

        b2r2_save = f'data/{dataset}/b2r2_6_pred.npy'
        if not os.path.exists(b2r2_save):
            maes_b2r2_6 = predict_CV(b2r2_6, barriers, CV=CV, mode='krr', save_hypers=False)
            np.save(b2r2_save, maes_b2r2_6)
        else:
            maes_b2r2_6 = np.load(b2r2_save)
        print(f'b2r2_6 mae {np.mean(maes_b2r2_6)} +- {np.std(maes_b2r2_6)}')
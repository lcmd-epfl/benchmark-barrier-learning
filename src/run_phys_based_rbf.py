import argparse as ap
from src.reaction_reps import QML, B2R2
from src.learning import predict_CV
import numpy as np
import os

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('-c', '--cyclo', action='store_true')
    parser.add_argument('-g', '--gdb', action='store_true')
    parser.add_argument('-p', '--proparg', action='store_true')
    parser.add_argument('-CV', '--CV', default=10)
    parser.add_argument('-tr', '--train', default=0.8)
    args = parser.parse_args()
    args.CV = int(args.CV)
    args.train = float(args.train)
    # maybe hydroform ?
    return args

if __name__ == "__main__":
    args = parse_args()
    cyclo = args.cyclo
    gdb = args.gdb
    proparg = args.proparg
    CV = args.CV

    if cyclo:
        print("Running for cyclo dataset")
        # 3d fingerprints SLATM
        qml = QML()
        qml.get_cyclo_data()
        barriers = qml.barriers
        slatm_save = 'data/cyclo/slatm.npy'
        if not os.path.exists(slatm_save):
            slatm = qml.get_SLATM()
            np.save(slatm_save, slatm)
        else:
            slatm = np.load(slatm_save)

        b2r2 = B2R2()
        b2r2.get_cyclo_data()
        b2r2_l_save = 'data/cyclo/b2r2_l.npy'
        if not os.path.exists(b2r2_l_save):
            b2r2_l = b2r2.get_b2r2_l()
            np.save(b2r2_l_save, b2r2_l)
        else:
            b2r2_l = np.load(b2r2_l_save)

        slatm_save = f'data/cyclo/slatm_{CV}_fold_rbf.npy'
        if not os.path.exists(slatm_save):
            maes_slatm = predict_CV(slatm, barriers, CV=CV, mode='krr', save_hypers=True, opt_kernels=['rbf'],
                                    save_file='data/cyclo/slatm_hypers.csv')
            np.save(slatm_save, maes_slatm)
        else:
            maes_slatm = np.load(slatm_save)
        print(f'slatm mae {np.mean(maes_slatm)} +- {np.std(maes_slatm)}')

        b2r2_l_save = f'data/cyclo/b2r2_l_{CV}_fold_rbf.npy'
        if not os.path.exists(b2r2_l_save):
            maes_b2r2_l = predict_CV(b2r2_l, barriers, CV=CV, mode='krr', save_hypers=True, opt_kernels='rbf',
                                     save_file='data/cyclo/b2r2_l_hypers.csv')
            np.save(b2r2_l_save, maes_b2r2_l)
        else:
            maes_b2r2_l = np.load(b2r2_l_save)
        print(f'b2r2_l mae {np.mean(maes_b2r2_l)} +- {np.std(maes_b2r2_l)}')

    if gdb:
        print("Running for gdb dataset")
        # 3d fingerprints SLATM
        qml = QML()
        qml.get_GDB7_ccsd_data()
        barriers = qml.barriers
        slatm_save = 'data/gdb7-22-ts/slatm.npy'
        if not os.path.exists(slatm_save):
            slatm = qml.get_SLATM()
            np.save(slatm_save, slatm)
        else:
            slatm = np.load(slatm_save)

        b2r2 = B2R2()
        b2r2.get_cyclo_data()
        b2r2_l_save = 'data/gdb7-22-ts/b2r2_l.npy'
        if not os.path.exists(b2r2_l_save):
            b2r2_l = b2r2.get_b2r2_l()
            np.save(b2r2_l_save, b2r2_l)
        else:
            b2r2_l = np.load(b2r2_l_save)

        slatm_save = f'data/gdb7-22-ts/slatm_{CV}_fold_rbf.npy'
        if not os.path.exists(slatm_save):
            maes_slatm = predict_CV(slatm, barriers, CV=CV, mode='krr', save_hypers=True, opt_kernels=['rbf'],
                                    save_file='data/cyclo/slatm_hypers.csv')
            np.save(slatm_save, maes_slatm)
        else:
            maes_slatm = np.load(slatm_save)
        print(f'slatm mae {np.mean(maes_slatm)} +- {np.std(maes_slatm)}')

        b2r2_l_save = f'data/gdb7-22-ts/b2r2_l_{CV}_fold_rbf.npy'
        if not os.path.exists(b2r2_l_save):
            maes_b2r2_l = predict_CV(b2r2_l, barriers, CV=CV, mode='krr', save_hypers=True, opt_kernels='rbf',
                                     save_file='data/cyclo/b2r2_l_hypers.csv')
            np.save(b2r2_l_save, maes_b2r2_l)
        else:
            maes_b2r2_l = np.load(b2r2_l_save)
        print(f'b2r2_l mae {np.mean(maes_b2r2_l)} +- {np.std(maes_b2r2_l)}')

    if proparg:
        print("Running for proparg dataset")
        # 3d fingerprints SLATM
        qml = QML()
        qml.get_proparg_data()
        barriers = qml.barriers
        slatm_save = 'data/proparg/slatm.npy'
        if not os.path.exists(slatm_save):
            slatm = qml.get_SLATM()
            np.save(slatm_save, slatm)
        else:
            slatm = np.load(slatm_save)

        b2r2 = B2R2()
        b2r2.get_proparg_data()
        b2r2_l_save = 'data/proparg/b2r2_l.npy'
        if not os.path.exists(b2r2_l_save):
            b2r2_l = b2r2.get_b2r2_l()
            np.save(b2r2_l_save, b2r2_l)
        else:
            b2r2_l = np.load(b2r2_l_save)

        slatm_save = f'data/proparg/slatm_{CV}_fold_rbf.npy'
        if not os.path.exists(slatm_save):
            maes_slatm = predict_CV(slatm, barriers, CV=CV, mode='krr', save_hypers=True, opt_kernels=['rbf'],
                                    save_file='data/proparg/slatm_hypers.csv')
            np.save(slatm_save, maes_slatm)
        else:
            maes_slatm = np.load(slatm_save)
        print(f'slatm mae {np.mean(maes_slatm)} +- {np.std(maes_slatm)}')

        b2r2_l_save = f'data/proparg/b2r2_l_{CV}_fold_rbf.npy'
        if not os.path.exists(b2r2_l_save):
            maes_b2r2_l = predict_CV(b2r2_l, barriers, CV=CV, mode='krr', save_hypers=True, opt_kernels='rbf',
                                     save_file='data/proparg/b2r2_l_hypers.csv')
            np.save(b2r2_l_save, maes_b2r2_l)
        else:
            maes_b2r2_l = np.load(b2r2_l_save)
        print(f'b2r2_l mae {np.mean(maes_b2r2_l)} +- {np.std(maes_b2r2_l)}')
import argparse as ap
from src.reaction_reps import TWODIM, QML
from src.learning import predict_CV_KRR, predict_CV_RF
import numpy as np
import os

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('-x', '--xtb', action='store_true', help='use xtb geom instead of dft geoms')
    parser.add_argument('-c', '--cyclo', action='store_true')
    parser.add_argument('-g', '--gdb', action='store_true')
    g = parser.add_mutually_exclusive_group()
    g.add_argument('-p', '--proparg', action='store_true')
    g.add_argument('--proparg_stereo', action='store_true')
    g.add_argument('--proparg_combinatorial', action='store_true')
    parser.add_argument('-CV', '--CV', default=10)
    parser.add_argument('-tr', '--train', default=0.8)
    args = parser.parse_args()
    args.CV = int(args.CV)
    args.train = float(args.train)
    return args

if __name__ == "__main__":
    args = parse_args()

    datasets = []
    if args.cyclo:
        datasets.append('cyclo')
    if args.gdb:
        datasets.append('gdb')
    if args.proparg or args.proparg_stereo or args.proparg_combinatorial:
        datasets.append('proparg')

    proparg_stereo = args.proparg_stereo
    proparg_combinatorial = args.proparg_combinatorial
    CV = args.CV

    for dataset in datasets:
        print(f"Running for {dataset} dataset")
        # first 2d fingerprints drfp, mfp
        if dataset == 'proparg':
            twodim = TWODIM(proparg_stereo=proparg_stereo, proparg_combinatorial=proparg_combinatorial)
            if proparg_stereo:
                drfp_save = 'data/proparg/drfp_stereo.npy'
            elif proparg_combinatorial:
                drfp_save = 'data/proparg/drfp_combinatorial.npy'
            else:
                drfp_save = 'data/proparg/drfp.npy'
            if proparg_stereo:
                mfp_save = 'data/proparg/mfp_stereo.npy'
            elif proparg_combinatorial:
                mfp_save = 'data/proparg/mfp_combinatorial.npy'
            else:
                mfp_save = 'data/proparg/mfp.npy'
        else:
            twodim = TWODIM()

            drfp_save = f'data/{dataset}/drfp.npy'
            mfp_save = f'data/{dataset}/mfp.npy'

        if not os.path.exists(drfp_save):
            if dataset == 'cyclo':
                drfp = twodim.get_cyclo_DRFP()
            elif dataset == 'gdb':
                drfp = twodim.get_gdb_DRFP()
            elif dataset == 'proparg':
                drfp = twodim.get_proparg_DRFP()
            np.save(drfp_save, drfp)
        else:
            drfp = np.load(drfp_save)

        if not os.path.exists(mfp_save):
            if dataset == 'cyclo':
                mfp = twodim.get_cyclo_MFP()
            elif dataset == 'gdb':
                mfp = twodim.get_gdb_MFP()
            elif dataset == 'proparg':
                mfp = twodim.get_proparg_MFP()
            np.save(mfp_save, mfp)
        else:
            mfp = np.load(mfp_save)

        barriers_twod = twodim.barriers

        # 3d fingerprints SLATM
        qml = QML()
        if args.xtb:
            if dataset == 'cyclo':
                qml.get_cyclo_xtb_data()
            elif dataset == 'proparg':
                qml.get_proparg_data_xtb()
            elif dataset == 'gdb':
                qml.get_GDB7_xtb_data()
            slatm_save = f'data/{dataset}/slatm_xtb.npy'
            b2r2_l_save = f'data/{dataset}/b2r2_l_xtb.npy'

        else:
            if dataset == 'cyclo':
                qml.get_cyclo_data()
            elif dataset == 'proparg':
                qml.get_proparg_data()
            elif dataset == 'gdb':
                qml.get_GDB7_ccsd_data()
            slatm_save = f'data/{dataset}/slatm.npy'
            b2r2_l_save = f'data/{dataset}/b2r2_l.npy'

        if not os.path.exists(slatm_save):
            slatm = qml.get_SLATM()
            np.save(slatm_save, slatm)
        else:
            slatm = np.load(slatm_save)

        if not os.path.exists(b2r2_l_save):
            b2r2_l = qml.get_b2r2_l()
            np.save(b2r2_l_save, b2r2_l)
        else:
            b2r2_l = np.load(b2r2_l_save)
        barriers_qml = qml.barriers

        print("reps generated/loaded, predicting")

        if dataset == 'proparg':
            if proparg_stereo:
                drfp_save = f'data/proparg/drfp_stereo_{CV}_fold.npy'
            elif proparg_combinatorial:
                drfp_save = f'data/proparg/drfp_combinatorial_{CV}_fold.npy'
            else:
                drfp_save = f'data/proparg/drfp_{CV}_fold.npy'

            if proparg_stereo:
                mfp_save = f'data/proparg/mfp_stereo_{CV}_fold.npy'
            elif proparg_combinatorial:
                mfp_save = f'data/proparg/mfp_combinatorial_{CV}_fold.npy'
            else:
                mfp_save = f'data/proparg/mfp_{CV}_fold.npy'

        else:
            drfp_save = f'data/{dataset}/drfp_{CV}_fold.npy'
            mfp_save = f'data/{dataset}/mfp_{CV}_fold.npy'

        if not os.path.exists(drfp_save):
            maes_drfp = predict_CV_RF(drfp, barriers_twod, CV=CV, train_size=0.8)
            np.save(drfp_save, maes_drfp)
        else:
            maes_drfp = np.load(drfp_save)
        print(f'drfp mae {np.mean(maes_drfp)} +- {np.std(maes_drfp)}')

        if not os.path.exists(mfp_save):
            maes_mfp = predict_CV_RF(mfp, barriers_twod, CV=CV, train_size=0.8)
            np.save(mfp_save, maes_mfp)
        else:
            maes_mfp = np.load(mfp_save)
        print(f'mfp mae {np.mean(maes_mfp)} +- {np.std(maes_mfp)}')

        if args.xtb:
            slatm_save = f"data/{dataset}/slatm_{CV}_fold_xtb.npy"
            b2r2_l_save = f"data/{dataset}/b2r2_l_{CV}_fold_xtb.npy"
            dataset_label = f'{dataset}_xtb'
        else:
            slatm_save = f'data/{dataset}/slatm_{CV}_fold.npy'
            b2r2_l_save = f'data/{dataset}/b2r2_l_{CV}_fold.npy'
            dataset_label = dataset

        if dataset == 'proparg':
            kernel = 'rbf'
        else:
            kernel = 'laplacian'

        if not os.path.exists(slatm_save):
            maes_slatm = predict_CV_KRR(slatm, barriers_qml, CV=CV, model='slatm', dataset=dataset_label)
            np.save(slatm_save, maes_slatm)
        else:
            maes_slatm = np.load(slatm_save)
        print(f'slatm mae {np.mean(maes_slatm)} +- {np.std(maes_slatm)}')

        if not os.path.exists(b2r2_l_save):
            maes_b2r2_l = predict_CV_KRR(b2r2_l, barriers_qml, CV=CV, model='b2r2', dataset=dataset_label)
            np.save(b2r2_l_save, maes_b2r2_l)
        else:
            maes_b2r2_l = np.load(b2r2_l_save)
        print(f'b2r2_l mae {np.mean(maes_b2r2_l)} +- {np.std(maes_b2r2_l)}')
import argparse as ap
from src.reaction_reps import TWODIM, QML, B2R2, SPAHM
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
        # first 2d fingerprints drfp, mfp
        twodim = TWODIM()
        drfp_save = 'data/cyclo/drfp.npy'
        if not os.path.exists(drfp_save):
            drfp = twodim.get_cyclo_DRFP()
            np.save(drfp_save, drfp)
        else:
            drfp = np.load(drfp_save)
        mfp_save = 'data/cyclo/mfp.npy'
        if not os.path.exists(mfp_save):
            mfp = twodim.get_cyclo_MFP()
            np.save(mfp_save, mfp)
        else:
            mfp = np.load(mfp_save)

        # 3d fingerprints SLATM and SPAHMb
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
        b2r2_save = 'data/cyclo/b2r2.npy'
        if not os.path.exists(b2r2_save):
            b2r2 = b2r2.get_b2r2_l()
            np.save(b2r2_save, b2r2)
        else:
            b2r2 = np.load(b2r2_save)

        # for now have to load spahm
        sp = SPAHM()
        sp.get_cyclo_data_and_rep()
        sp_barriers = sp.barriers
        spahm = sp.spahm_b

        print("reps generated/loaded, predicting")

        drfp_save = f'data/cyclo/drfp_{CV}_fold.npy'
        if not os.path.exists(drfp_save):
            maes_drfp = predict_CV(drfp, barriers, CV=CV, mode='rf')
            np.save(drfp_save, maes_drfp)
        else:
            maes_drfp = np.load(drfp_save)
        print(f'drfp mae {np.mean(maes_drfp)} +- {np.std(maes_drfp)}')

        mfp_save = f'data/cyclo/mfp_{CV}_fold.npy'
        if not os.path.exists(mfp_save):
            maes_mfp = predict_CV(mfp, barriers, CV=CV, mode='rf')
            np.save(mfp_save, maes_mfp)
        else:
            maes_mfp = np.load(mfp_save)
        print(f'mfp mae {np.mean(maes_mfp)} +- {np.std(maes_mfp)}')

        slatm_save = f'data/cyclo/slatm_{CV}_fold.npy'
        if not os.path.exists(slatm_save):
            maes_slatm = predict_CV(slatm, barriers, CV=CV, mode='krr', kernel='rbf')
            np.save(slatm_save, maes_slatm)
        else:
            maes_slatm = np.load(slatm_save)
        print(f'slatm mae {np.mean(maes_slatm)} +- {np.std(maes_slatm)}')

        spahm_save = f'data/cyclo/spahm_{CV}_fold.npy'
        if not os.path.exists(spahm_save):
            maes_spahm = predict_CV(spahm, sp_barriers, CV=CV, mode='krr', kernel='rbf')
            np.save(spahm_save, maes_spahm)
        else:
            maes_spahm = np.load(spahm_save)
        print(f'spahm mae {np.mean(maes_spahm)} +- {np.std(maes_spahm)}')

        b2r2_save = f'data/cyclo/b2r2_{CV}_fold.npy'
        if not os.path.exists(b2r2_save):
            maes_b2r2 = predict_CV(b2r2, barriers, CV=CV, mode='krr', kernel='rbf')
            np.save(b2r2_save, maes_b2r2)
        else:
            maes_b2r2 = np.load(b2r2_save)
        print(f'b2r2 mae {np.mean(maes_b2r2)} +- {np.std(maes_b2r2)}')

    if gdb:
        print("Running for gdb dataset")
        # first 2d fingerprints drfp, mfp
        twodim = TWODIM()
        drfp_save = 'data/gdb7-22-ts/drfp.npy'
        if not os.path.exists(drfp_save):
            drfp = twodim.get_gdb_DRFP()
            np.save(drfp_save, drfp)
        else:
            drfp = np.load(drfp_save)
        mfp_save = 'data/gdb7-22-ts/mfp.npy'
        if not os.path.exists(mfp_save):
            mfp = twodim.get_gdb_MFP()
            np.save(mfp_save, mfp)
        else:
            mfp = np.load(mfp_save)

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

        b2r2 = B2R2()
        b2r2.get_GDB7_ccsd_data()
        b2r2_save = 'data/gdb7-22-ts/b2r2.npy'
        if not os.path.exists(b2r2_save):
            b2r2 = b2r2.get_b2r2_l()
            np.save(b2r2_save, b2r2)
        else:
            b2r2 = np.load(b2r2_save)

        # for now have to load spahm
        sp = SPAHM()
        sp.get_gdb_data_and_rep()
        spahm = sp.spahm_b
        sp_barriers = sp.barriers

        print("reps generated/loaded, predicting")

        drfp_save = f'data/gdb7-22-ts/drfp_{CV}_fold.npy'
        if not os.path.exists(drfp_save):
            maes_drfp = predict_CV(drfp, barriers, CV=CV, mode='rf')
            np.save(drfp_save, maes_drfp)
        else:
            maes_drfp = np.load(drfp_save)
        print(f'drfp mae {np.mean(maes_drfp)} +- {np.std(maes_drfp)}')

        mfp_save = f'data/gdb7-22-ts/mfp_{CV}_fold.npy'
        if not os.path.exists(mfp_save):
            maes_mfp = predict_CV(mfp, barriers, CV=CV, mode='rf')
            np.save(mfp_save, maes_mfp)
        else:
            maes_mfp = np.load(mfp_save)
        print(f'mfp mae {np.mean(maes_mfp)} +- {np.std(maes_mfp)}')

        slatm_save = f'data/gdb7-22-ts/slatm_{CV}_fold.npy'
        if not os.path.exists(slatm_save):
            maes_slatm = predict_CV(slatm, barriers, CV=CV, mode='krr', kernel='rbf')
            np.save(slatm_save, maes_slatm)
        else:
            maes_slatm = np.load(slatm_save)
        print(f'slatm mae {np.mean(maes_slatm)} +- {np.std(maes_slatm)}')

        spahm_save = f'data/gdb7-22-ts/spahm_{CV}_fold.npy'
        if not os.path.exists(spahm_save):
            maes_spahm = predict_CV(spahm, sp_barriers, CV=CV, mode='krr', kernel='laplacian')
            np.save(spahm_save, maes_spahm)
        else:
            maes_spahm = np.load(spahm_save)
        print(f'spahm mae {np.mean(maes_spahm)} +- {np.std(maes_spahm)}')

        b2r2_save = f'data/gdb7-22-ts/b2r2_{CV}_fold.npy'
        if not os.path.exists(b2r2_save):
            maes_b2r2 = predict_CV(b2r2, barriers, CV=CV, mode='krr', kernel='rbf')
            np.save(b2r2_save, maes_b2r2)
        else:
            maes_b2r2 = np.load(b2r2_save)
        print(f'b2r2 mae {np.mean(maes_b2r2)} +- {np.std(maes_b2r2)}')

    if proparg:
        print("Running for proparg dataset")
        print("PROBLEMS WITH CSV FILES HERE. NEED TO CHECK SMILES.")
        # first 2d fingerprints drfp, mfp
        twodim = TWODIM()
        drfp_save = 'data/proparg/drfp.npy'
        if not os.path.exists(drfp_save):
            drfp = twodim.get_proparg_DRFP()
            np.save(drfp_save, drfp)
        else:
            drfp = np.load(drfp_save)
        mfp_save = 'data/proparg/mfp.npy'
        if not os.path.exists(mfp_save):
            mfp = twodim.get_proparg_MFP()
            np.save(mfp_save, mfp)
        else:
            mfp = np.load(mfp_save)

        # 3d fingerprints SLATM and SPAHMb
        qml = QML()
        qml.get_proparg_data()
        slatm_save = 'data/proparg/slatm.npy'
        if not os.path.exists(slatm_save):
            slatm = qml.get_SLATM()
            np.save(slatm_save, slatm)
        else:
            slatm = np.load(slatm_save)

        barriers = qml.barriers
        #print('barriers shape', barriers.shape)

        b2r2 = B2R2()
        b2r2.get_proparg_data()
        b2r2_save = 'data/proparg/b2r2.npy'
        if not os.path.exists(b2r2_save):
            b2r2 = b2r2.get_b2r2_l()
            np.save(b2r2_save, b2r2)
        else:
            b2r2 = np.load(b2r2_save)

        # for now have to load spahm
        sp = SPAHM()
        sp.get_proparg_data_and_rep()
        sp_barriers = sp.barriers

        print("reps generated/loaded, predicting")

        drfp_save = f'data/proparg/drfp_{CV}_fold.npy'
        if not os.path.exists(drfp_save):
            maes_drfp = predict_CV(drfp, barriers, CV=CV, mode='rf')
            np.save(drfp_save, maes_drfp)
        else:
            maes_drfp = np.load(drfp_save)
        print(f'drfp mae {np.mean(maes_drfp)} +- {np.std(maes_drfp)}')

        mfp_save = f'data/proparg/drfp_{CV}_fold.npy'
        if not os.path.exists(mfp_save):
            maes_mfp = predict_CV(mfp, barriers, CV=CV, mode='rf')
            np.save(mfp_save, maes_mfp)
        else:
            maes_mfp = np.load(mfp_save)
        print(f'mfp mae {np.mean(maes_mfp)} +- {np.std(maes_mfp)}')

        slatm_save = f'data/proparg/slatm_{CV}_fold.npy'
        if not os.path.exists(slatm_save):
            maes_slatm = predict_CV(slatm, barriers, CV=CV, mode='krr', kernel='rbf')
            np.save(slatm_save, maes_slatm)
        else:
            maes_slatm = np.load(slatm_save)
        print(f'slatm mae {np.mean(maes_slatm)} +- {np.std(maes_slatm)}')

        spahm_save = f'data/proparg/spahm_{CV}_fold.npy'
        if not os.path.exists(spahm_save):
            maes_spahm = predict_CV(spahm, barriers, CV=CV, mode='krr', kernel='rbf')
            np.save(spahm_save, maes_spahm)
        else:
            maes_spahm = np.load(spahm_save)
        print(f'spahm mae {np.mean(maes_spahm)} +- {np.std(maes_spahm)}')

        b2r2_save = f'data/proparg/b2r2_{CV}_fold.npy'
        if not os.path.exists(b2r2_save):
            maes_b2r2 = predict_CV(b2r2, barriers, CV=CV, mode='krr', kernel='rbf')
            np.save(b2r2_save, maes_b2r2)
        else:
            maes_b2r2 = np.load(b2r2_save)
        print(f'b2r2 mae {np.mean(maes_b2r2)} +- {np.std(maes_b2r2)}')

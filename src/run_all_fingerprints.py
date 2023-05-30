import argparse as ap
from src.reaction_reps import TWODIM, QML, B2R2
from src.learning import predict_CV
import numpy as np
import os
def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('-c', '--cyclo', action='store_true')
    parser.add_argument('-g', '--gdb', action='store_true')
    parser.add_argument('-p', '--proparg', action='store_true')
    parser.add_argument('-CV', '--CV', default=1)
    args = parser.parse_args()
    args.CV = int(args.CV)
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
            print('drfp shape', drfp.shape)
            np.save(drfp_save, drfp)
        else:
            drfp = np.load(drfp_save)
        mfp_save = 'data/cyclo/mfp.npy'
        if not os.path.exists(mfp_save):
            mfp = twodim.get_cyclo_MFP()
            print('mfp shape', mfp.shape)
            np.save(mfp_save, mfp)
        else:
            mfp = np.load(mfp_save)

        barriers = twodim.barriers

        print("reps generated/loaded, predicting")

        drfp_save = f'data/cyclo/drfp_{CV}_fold.npy'
        if not os.path.exists(drfp_save):
            maes_drfp = predict_CV(drfp, barriers, CV=CV, mode='rf')
        else:
            maes_drfp = np.load(drfp_save)
        print(f'drfp mae {np.mean(maes_drfp)} +- {np.std(maes_drfp)}')

        mfp_save = f'data/cyclo/mfp_{CV}_fold.npy'
        if not os.path.exists(mfp_save):
            maes_mfp = predict_CV(mfp, barriers, CV=CV, mode='rf')
        else:
            maes_drfp = np.load(mfp_save)
        print(f'mfp mae {np.mean(maes_mfp)} +- {np.std(maes_mfp)}')

        # 3d fingerprints SLATM and SPAHMb




import argparse as ap
from src.reaction_reps import TWODIM, QML
from src.learning import predict_CV_KRR, predict_CV_RF
import numpy as np
import os
from timeit import default_timer as timer

if __name__ == "__main__":

    datasets = ['proparg', 'cyclo', 'gdb']
    subset = 750

    for i, dataset in enumerate(datasets):
        print(f"Running for {dataset} dataset")
        twodim = TWODIM()

        # DRFP
        start_data = timer()
        if dataset == 'cyclo':
            drfp = twodim.get_cyclo_DRFP(subset=subset)
        elif dataset == 'gdb':
            drfp = twodim.get_gdb_DRFP(subset=subset)
        elif dataset == 'proparg':
            drfp = twodim.get_proparg_DRFP(subset=subset)
        end_data = timer()
        rep_gen_time_drfp = end_data - start_data
        print(f"{rep_gen_time_drfp=}")

        # MFP
        start_data = timer()
        if dataset == 'cyclo':
            mfp = twodim.get_cyclo_MFP(subset=subset)
        elif dataset == 'gdb':
            mfp = twodim.get_gdb_MFP(subset=subset)
        elif dataset == 'proparg':
            mfp = twodim.get_proparg_MFP(subset=subset)
        end_data = timer()
        rep_gen_time_mfp = end_data - start_data
        print(f"{rep_gen_time_mfp=}")

        barriers_twod = twodim.barriers

        # 3d fingerprints SLATM
        qml = QML()
        if dataset == 'cyclo':
            qml.get_cyclo_data(subset=subset)
        elif dataset == 'proparg':
            qml.get_proparg_data(subset=subset)
        elif dataset == 'gdb':
            qml.get_GDB7_ccsd_data(subset=subset)

        #SLATM
        start_data = timer()
        slatm = qml.get_SLATM()
        end_data = timer()
        rep_gen_time_slatm = end_data - start_data
        print(f"{rep_gen_time_slatm=}")

        # B2R2
        start_data = timer()
        b2r2_l = qml.get_b2r2_l()
        end_data = timer()
        rep_gen_time_b2r2 = end_data - start_data
        print(f"{rep_gen_time_b2r2=}")

        barriers_qml = qml.barriers

        _, train_time_drfp, pred_time_drfp = predict_CV_RF(drfp, barriers_twod, CV=1, train_size=0.8, model='drfp', dataset=dataset, timing=True)
        print(f"{train_time_drfp=}, {pred_time_drfp=}")

        _, train_time_mfp, pred_time_mfp = predict_CV_RF(mfp, barriers_twod, CV=1, train_size=0.8, model='mfp', dataset=dataset, timing=True)
        print(f"{train_time_mfp=}, {pred_time_mfp=}")

        _, train_time_slatm, pred_time_slatm = predict_CV_KRR(slatm, barriers_qml, CV=1, train_size=0.8, model='slatm', dataset=dataset, timing=True)
        print(f"{train_time_slatm=}, {pred_time_slatm=}")

        _, train_time_b2r2, pred_time_b2r2 = predict_CV_KRR(b2r2_l, barriers_qml, CV=1, train_size=0.8, model='b2r2', dataset=dataset, timing=True)
        print(f"{train_time_b2r2=}, {pred_time_b2r2=}")

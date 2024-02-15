#!/usr/bin/env python3

import os
import numpy as np
from src.reaction_reps import QML


def test_slatm():
    qml = QML()
    qml.get_GDB7_ccsd_data(subset=32)
    slatm = qml.get_SLATM()
    slatm_0 = np.load(f'{os.path.dirname(os.path.realpath(__file__))}/data/slatm_d.npy')
    assert(np.linalg.norm(slatm-slatm_0) < 1e-10)


if __name__ == "__main__":
    test_slatm()

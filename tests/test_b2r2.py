#!/usr/bin/env python3

import os
import numpy as np
from src.reaction_reps import QML


def test_b2r2_l():
    _test_b2r2('l')
def test_b2r2_a():
    _test_b2r2('a')
def test_b2r2_n():
    _test_b2r2('n')


def _test_b2r2(variant):
    qml = QML()
    qml.get_GDB7_ccsd_data(subset=32)
    try:  # new version
        b2r2 = qml.get_b2r2(variant=variant)
    except:  # old version
        b2r2 = {'n': qml.get_b2r2_n, 'l': qml.get_b2r2_l, 'a': qml.get_b2r2_a}[variant]()
    b2r2_0 = np.load(f'{os.path.dirname(os.path.realpath(__file__))}/data/b2r2_{variant}.npy')
    assert(np.linalg.norm(b2r2-b2r2_0) < 1e-10)


if __name__ == "__main__":
    test_b2r2_l()
    test_b2r2_a()
    test_b2r2_n()

HYPERS_SLATM = {
"cyclo" : ['laplacian', 0.001,0.0001],
"cyclo_xtb" : ['laplacian', 0.0001, 0.0001],
"gdb" : ['laplacian', 0.01,0.0001],
"gdb_xtb" : ['laplacian', 0.01, 0.0001],
"proparg" : ['rbf', 1,1e-10],
"proparg_xtb" : ['laplacian', 1e-5, 1e-10]
}

HYPERS_B2R2 = {
    "cyclo": ['laplacian', 0.0001, 1e-10],
    "gdb": ['laplacian', 0.0001, 0.0001],
    "proparg": ['laplacian', 1e-5, 1e-10],
    "gdb_xtb": ['laplacian', 0.0001, 0.0001],
    "cyclo_xtb": ['laplacian', 0.0001, 0.0001],
    "proparg_xtb": ['laplacian', 1e-5, 0.0001],
}

HYPERS_DRFP = {}

HYPERS_MFP = {}
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

HYPERS_DRFP = {
    "gdb": {'bootstrap': False, 'max_depth': 90, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300},
    "cyclo": {'bootstrap': False, 'max_depth': 60, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200},
    "proparg" : {'bootstrap': False, 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
}

HYPERS_MFP = {
    "gdb": {'bootstrap': False, 'max_depth': 90, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100},
    "cyclo": {'bootstrap': False, 'max_depth': 90, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100},
    "proparg" : {'bootstrap': False, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
}
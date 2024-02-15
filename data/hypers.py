HYPERS_SLATM = {
             "cyclo" : ['laplacian', 0.001, 1e-10],
                 "cyclo_xtb" : ['rbf', 10, 1e-10],
                     "gdb" : ['laplacian', 0.01, 1e-10],
                         "gdb_xtb": ['laplacian', 0.01, 0.0001],
                             "proparg_xtb" : ['rbf', 1000, 1e-4],
                                 "proparg" : ['rbf', 10000, 1e-10]
                                }
HYPERS_SLATM['gdb7-22-ts_xtb'] = HYPERS_SLATM['gdb_xtb']
HYPERS_SLATM['']

HYPERS_B2R2 = {
             "cyclo" : ['laplacian', 0.0001, 1e-10],
                 "cyclo_xtb" : ['rbf', 100, 0.0001],
                     "gdb" : ['laplacian', 0.0001, 0.0001],
                         "gdb_xtb": ['laplacian', 0.0001, 0.0001],
                            "proparg_xtb": ["rbf", 1000, 0.0001],
                                 "proparg" : ['laplacian', 1e-5, 1e-10]
                                }
HYPERS_B2R2['gdb7-22-ts_xtb'] = HYPERS_B2R2['gdb_xtb']

HYPERS_DRFP = {
            "cyclo": {'bootstrap': False, 'max_depth': 80, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 294},
                "gdb": {'bootstrap': False, 'max_depth': 70, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 172},
                    "proparg":{'bootstrap': False, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 50}
                    }

HYPERS_MFP = {
            "cyclo": {'bootstrap': False, 'max_depth': 50, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 233},
                'gdb': {'bootstrap': False, 'max_depth': 90, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 355},
                    "proparg" :{'bootstrap': False, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 477}
                    }

HYPERS_LANG = {
            "proparg" : {"lr":1e-3, "p":0.8},
                "cyclo" : {'lr':1e-4, "p":0.2},
                    "gdb" : {'lr':1e-4, "p":0.2}
                    }


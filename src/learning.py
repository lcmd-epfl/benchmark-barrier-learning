import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split, KFold
from scipy.spatial import distance_matrix
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from src.hypers import *
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from timeit import default_timer as timer

def get_mae(y_pred, y_test):
    return np.mean(np.abs(y_test - y_pred))
def normalize_data(y_train, y_test):
    mean = np.mean(y_train)
    std = np.mean(y_train)
    y_train = (y_train - mean) / std
    y_test = (y_test - mean) / std
    return y_train, y_test, mean, std

def unnormalize_data(data, mean, std):
    return data * std + mean

def predict_laplacian_KRR(D_train, D_test,
                y_train, y_test,
                gamma=0.001, l2reg=1e-10,
                          timing=False):
    """Perform KRR and return MAE of prediction using laplacian kernel.

    Args:
        D_train (np array): Distance matrix for the training data
        D_test (np array): Distance matrix between training and out-of-sample
        y_train (np array): Training labels
        y_test (np array): Labels for out-of-sample prediction
        l2reg (float): Regularizer. Default 1e-10
        gamma (float): gamma value for laplacian kernel. Default 1e-3

    Returns:
        float: Mean Absolute Error of prediction
    """
    start_train = timer()
    K      = np.exp(-gamma*D_train)
    K[np.diag_indices_from(K)] += l2reg
    alpha = np.dot(np.linalg.inv(K), y_train)
    end_train = timer()
    partial_train = end_train - start_train

    start_test = timer()
    K_test = np.exp(-gamma*D_test)
    y_pred = np.dot(K_test, alpha)
    end_test = timer()
    partial_test = end_test - end_train
    if timing:
        return y_pred, partial_train, partial_test
    return y_pred

def predict_gaussian_KRR(D_train, D_test,
                        y_train, y_test,
                        sigma=100, l2reg=1e-10):
    """
    Now for gaussian kernel
    """
    K      = np.exp(-D_train / (2*sigma**2))
    K_test = np.exp(-D_test / (2*sigma**2))
    K[np.diag_indices_from(K)] += l2reg
    alpha = np.dot(np.linalg.inv(K), y_train)

    y_pred = np.dot(K_test, alpha)
    return y_pred

def opt_hyperparams_w_kernel(
        X, y,
        idx_train,
        idx_val,
        gammas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
        l2regs=[1e-10, 1e-7, 1e-4],
        sigmas=[1, 10, 100, 1e3, 1e4],
):
    print(f"Hyperparam search including kernels")
    D_full_laplace = pairwise_distances(X, metric='l1')
    D_full_rbf = pairwise_distances(X, metric='l2')
    for D_full in [D_full_rbf, D_full_laplace]:
        D_train = D_full[np.ix_(idx_train, idx_train)]
        D_val = D_full[np.ix_(idx_val, idx_train)]
        y_train = y[idx_train]
        y_val = y[idx_val]

    sigma, l2reg_g, mae_g = opt_hyperparams_gaussian(D_train, D_val, y_train, y_val, sigmas=sigmas, l2regs=l2regs)
    gamma, l2reg_l, mae_l = opt_hyperparams_laplacian(D_train, D_val, y_train, y_val, gammas=gammas, l2regs=l2regs)
    if mae_g < mae_l:
        return ('rbf', sigma, l2reg_g, D_full_rbf)
    else:
        return ('laplacian', gamma, l2reg_l, D_full_laplace)

def opt_hyperparams_laplacian(
    D_train, D_val,
    y_train, y_val,
    gammas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
    l2regs = [1e-10, 1e-7, 1e-4],
    mean=0,
    std=1,
):
    """Optimize hyperparameters for KRR with
    laplacian kernel.
    Assuming input data is normalized, providing mean and std
    """

    print("Hyperparam search for laplacian kernel")
     # Laplacian
    maes_lap = np.zeros((len(gammas), len(l2regs)))

    for i, gamma in enumerate(gammas):
        for j, l2reg in enumerate(l2regs):
            y_pred = predict_laplacian_KRR(
                D_train, D_val, y_train, y_val, gamma=gamma, l2reg=l2reg
                )
            y_pred = unnormalize_data(y_pred, mean, std)
            y_val = unnormalize_data(y_val, mean, std)
            mae = get_mae(y_pred, y_val)
            print(f'{mae=} for params {gamma=} {l2reg=}')
            maes_lap[i, j] = mae
    min_i, min_j = np.unravel_index(np.argmin(maes_lap, axis=None), maes_lap.shape)
    min_gamma = gammas[min_i]
    min_l2reg = l2regs[min_j]
    min_mae_lap = maes_lap[min_i, min_j]

    print(f"Best mae={min_mae_lap} for gamma={min_gamma} and l2reg={min_l2reg}")

    return min_gamma, min_l2reg, min_mae_lap

def opt_hyperparams_gaussian(
    D_train, D_val,
    y_train, y_val,
    sigmas = [1, 10, 100, 1e3, 1e4],
    l2regs = [1e-10, 1e-7, 1e-4],
    mean = 0,
    std = 1,
):
    """Optimize hyperparameters for KRR with
    gaussian kernel.
    Assuming data is normalized, providing mean and std to un-normalize
    """

    print("Hyperparam search for gaussian kernel")
    maes = np.zeros((len(sigmas), len(l2regs)))

    for i, sigma in enumerate(sigmas):
        for j, l2reg in enumerate(l2regs):
            y_pred = predict_gaussian_KRR(
                D_train, D_val, y_train, y_val, sigma=sigma, l2reg=l2reg
                )
            y_pred = unnormalize_data(y_pred, mean, std)
            y_val = unnormalize_data(y_val, mean, std)
            mae = get_mae(y_pred, y_val)
            print(f'{mae=} for params {sigma=} {l2reg=}')
            maes[i, j] = mae
    min_i, min_j = np.unravel_index(np.argmin(maes, axis=None), maes.shape)
    min_sigma = sigmas[min_i]
    min_l2reg = l2regs[min_j]
    min_mae = maes[min_i, min_j]

    print(f"Best mae={min_mae} for sigma={min_sigma} and l2reg={min_l2reg}")

    return min_sigma, min_l2reg, min_mae

def CV_KRR_optional_hyperopt(X, y, seed=1, CV=10, opt=False, kernel='',
           sig_gam=None, l2reg=None, train_size=0.8, test_size=None, timing=False):
    maes = np.zeros((CV))

    for i in range(CV):
        seed += i

        idx_train, idx_test_val = train_test_split(np.arange(len(y)), random_state=seed, train_size=train_size,
                                                   test_size=None)
        idx_test, idx_val = train_test_split(idx_test_val, shuffle=False, test_size=0.5)

        if i == 0:
            if opt:
                # hyperparam opt
                print("Optimising hypers...")
                kernel, sig_gam, l2reg, D_full = opt_hyperparams_w_kernel(X, y, idx_train, idx_val)
                print(f"Using opt hypers kernel: {kernel} sig_gam: {sig_gam} l2reg: {l2reg}")

            # use params and compute D_full
            if kernel == 'laplacian':
                D_full = pairwise_distances(X, metric='l1')
            elif kernel == 'rbf' or kernel == 'gaussian':
                D_full = pairwise_distances(X, metric='l2')
            else:
                raise NotImplementedError(f"kernel {kernel} is not implemented")

        start_train = timer()
        D_train = D_full[np.ix_(idx_train, idx_train)]
        end_train = timer()
        partial_train_ = end_train - start_train

        y_train = y[idx_train]

        y_val   = y[idx_val]
        start_test = timer()
        D_test  = D_full[np.ix_(idx_test,  idx_train)]
        end_test = timer()
        partial_test_ = end_test - start_test
        y_test  = y[idx_test]

        y_train, y_test, mean, std = normalize_data(y_train, y_test)

        if kernel == 'laplacian':
            out = predict_laplacian_KRR(D_train, D_test,
                                 y_train, y_test,
                                 l2reg=l2reg, gamma=sig_gam,
                                           timing=timing)
            if timing:
                y_pred, partial_train, partial_test = out
            else:
                y_pred = out
            y_pred = unnormalize_data(y_pred, mean, std)
            y_test = unnormalize_data(y_test, mean, std)
            mae = get_mae(y_pred, y_test)
        elif kernel == 'rbf' or kernel == 'gaussian':
            y_pred = predict_gaussian_KRR(D_train, D_test,
                                 y_train, y_test,
                                 l2reg=l2reg, sigma=sig_gam)
            y_pred = unnormalize_data(y_pred, mean, std)
            y_test = unnormalize_data(y_test, mean, std)
            mae = get_mae(y_pred, y_test)

        maes[i] = mae
    if timing:
        train_time = partial_train + partial_train_
        test_time = partial_test + partial_test_
        return maes, train_time, test_time
    return maes

def predict_CV_KRR(X, y, CV=10, seed=1, train_size=0.8, test_size=None,
               dataset='', model='', timing=False):
    if model == 'slatm':
        HYPERS = HYPERS_SLATM
    elif model == 'b2r2':
        HYPERS = HYPERS_B2R2
    else:
        raise NotImplementedError("CV is not implemented here for methods other than slatm, b2r2")

    if dataset in HYPERS.keys():
        kernel, sig_gam, l2reg = HYPERS[dataset]
        print(f"Hypers for {model} read from file. Kernel: {kernel}, sigma/gamma: {sig_gam}, l2reg: {l2reg}")
        maes = CV_KRR_optional_hyperopt(X, y, seed=seed, CV=CV, train_size=train_size, test_size=test_size, kernel=kernel,
                                        sig_gam=sig_gam, l2reg=l2reg, opt=False, timing=timing)
        if timing:
            maes, train_time, test_time = maes
            return maes, train_time, test_time
    else:
        print(f"Optimizing hypers...")
        maes = CV_KRR_optional_hyperopt(X,y, seed=seed, CV=CV, train_size=train_size, test_size=test_size, opt=True, timing=timing)

    return maes

def predict_RF(X_train, X_test, y_train, y_test, seed=1,
               n_estimators=100, max_depth=None, max_features='auto',
               min_samples_split=2, min_samples_leaf=1,
               bootstrap=True, timing=False):
    rf = RandomForestRegressor(random_state=seed,
                               n_estimators=n_estimators,
                               max_depth=max_depth,
                               max_features=max_features,
                               min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf,
                               bootstrap=bootstrap
                               )
    if timing:
        start_train = timer()
    rf.fit(X_train, y_train)
    if timing:
        end_train = timer()
        train_time = end_train - start_train
        start_pred = timer()
    y_pred = rf.predict(X_test)
    if timing:
        end_pred = timer()
        pred_time = end_pred - start_pred
    if timing:
        return y_pred, train_time, pred_time
    return y_pred

def predict_CV_RF(X, y, CV=10, seed=1, train_size=0.8, test_size=None, dataset='', model='', timing=False):
    maes = np.zeros((CV))
    if model == 'mfp':
        HYPERS = HYPERS_MFP
    elif model == 'drfp':
        HYPERS = HYPERS_DRFP

    for i in range(CV):
        seed += i
        X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, random_state=seed, train_size=train_size,
                                                                    test_size=test_size)
        y_train, y_test_val, mean, std = normalize_data(y_train, y_test_val)

        X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, shuffle=False, train_size=0.5)

        if i == 0:
            if dataset in HYPERS.keys():
                print(f"Using hypers from file...")
                best = HYPERS[dataset]
            else:
                print(f"Optimising hypers...")
                def tune_hypers_RF(space):
                    model = RandomForestRegressor(random_state=space['seed'],
                                                  max_depth=space['max_depth'],
                                                  n_estimators=space['n_estimators'],
                                                  max_features=space["max_features"],
                                                  min_samples_split=space["min_samples_split"],
                                                  min_samples_leaf=space["min_samples_leaf"],
                                                  bootstrap=space["bootstrap"],
                                                  )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred = unnormalize_data(y_pred, mean, std)
                    y_test_ = unnormalize_data(y_test, mean, std)
                    mae = get_mae(y_pred, y_test_)
                    return {"loss": mae, "status": STATUS_OK, "model": model}
                space = {'max_depth': hp.choice("max_depth", np.linspace(10,100,10, dtype=int)),
                         'n_estimators': hp.choice('n_estimators', np.linspace(100, 600, 10, dtype=int)),
                         'max_features': hp.choice('max_features', ['log2', 'sqrt']),
                         'min_samples_split': hp.choice('min_samples_split', [2,5,10]),
                         'min_samples_leaf': hp.choice('min_samples_leaf', [1,2,4]),
                         'bootstrap': hp.choice('bootstrap', [True, False]),
                         'seed' : 1,
                         }
                trials = Trials()
                best = fmin(fn=tune_hypers_RF,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=100,
                            trials=trials)
                # for hp choice will return index
                best['max_depth'] = np.linspace(10,100,10, dtype=int)[best['max_depth']]
                best['n_estimators'] = np.linspace(50, 600, 10, dtype=int)[best['n_estimators']]
                best['max_features'] = np.array(['log2', 'sqrt'])[best['max_features']]
                best['min_samples_split'] = np.array([2,5,10])[best['min_samples_split']]
                best['min_samples_leaf'] = np.array([1,2,4])[best['min_samples_leaf']]
                best['bootstrap'] = np.array([True, False])[best['bootstrap']]
            print(f'using opt hyperparams {best=}') # dictionary of params

        out = predict_RF(X_train, X_test, y_train, y_test, max_depth=best['max_depth'],
                            n_estimators=best['n_estimators'], max_features=best['max_features'],
                            min_samples_split=best['min_samples_split'], min_samples_leaf=best['min_samples_leaf'],
                            bootstrap=best['bootstrap'], seed=1, timing=timing)
        if timing:
            y_pred, train_time, pred_time = out
        else:
            y_pred = out

        y_pred = unnormalize_data(y_pred, mean, std)
        y_test = unnormalize_data(y_test, mean, std)
        mae = get_mae(y_pred, y_test)
        maes[i] = mae
    if timing:
        return maes, train_time, pred_time
    return maes

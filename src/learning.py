import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from sklearn.model_selection import train_test_split, KFold
from scipy.spatial import distance_matrix
import xgboost as xgb
import pandas as pd

def predict_RF(X_train, X_test, y_train, y_test):
    rf = xgb.XGBRegressor()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mae = np.mean(np.abs(y_test - y_pred))
    return mae, y_pred


def predict_KRR(X_train, X_test, y_train, y_test, sigma=100, l2reg=1e-6, gamma=0.001,
        kernel='rbf'):
    """Perform KRR and return MAE of prediction.

    Args:
        X_train (np array): Training data
        X_test (np array): Data for out-of-sample prediction
        y_train (np array): Training labels
        y_test (np array): Labels for out-of-sample prediction
        sigma (int): Kernel width. Default 100
        l2reg (float): Regularizer. Default 1e-6
        gamma (float): gamma value for laplacian kernel. Default 1e-3
        kernel (str): whether kernel is rbf / gaussian, laplacian or polynomial

    Returns:
        float: Mean Absolute Error of prediction
    """

    g_gauss = 1.0 / (2 * sigma ** 2)

    if kernel=='rbf' or kernel=='gaussian':
        K = rbf_kernel(X_train, X_train, gamma=g_gauss)
        K[np.diag_indices_from(K)] += l2reg
        alpha = np.dot(np.linalg.inv(K), y_train)
        K_test = rbf_kernel(X_test, X_train, gamma=g_gauss)
    elif kernel == 'laplacian':
        K = laplacian_kernel(X_train, X_train, gamma=gamma)
        K[np.diag_indices_from(K)] += l2reg
        alpha = np.dot(np.linalg.inv(K), y_train)
        K_test = laplacian_kernel(X_test, X_train, gamma=gamma)
    else:
        print('kernel is not specified')

    y_pred = np.dot(K_test, alpha)
    mae = np.mean(np.abs(y_test - y_pred))
    return mae, y_pred


def opt_hyperparams(
    X_train, X_val, y_train, y_val,
     sigmas = [1,10,100,1000],
    gammas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
    l2regs = [1e-10, 1e-7, 1e-4],
):
    """Optimize hyperparameters for KRR with gaussian or 
    laplacian kernel.
    """

    # RBF
    kernel = 'rbf'
    maes_rbf = np.zeros((len(sigmas), len(l2regs)))
    for i, sigma in enumerate(sigmas):
        for j, l2reg in enumerate(l2regs):
            mae, y_pred = predict_KRR(
                    X_train, X_val, y_train, y_val, sigma=sigma, l2reg=l2reg, kernel=kernel
                    )
         #   print(f'mae={mae} for params {kernel, sigma, l2reg}')
            maes_rbf[i, j] = mae
    min_i, min_j = np.unravel_index(np.argmin(maes_rbf, axis=None), maes_rbf.shape)
    min_sigma = sigmas[min_i]
    min_l2reg_rbf = l2regs[min_j]
    min_mae_rbf = maes_rbf[min_i, min_j]

    # Laplacian
    kernel = 'laplacian'
    maes_lap = np.zeros((len(gammas), len(l2regs)))
    for i, gamma in enumerate(gammas):
        for j, l2reg in enumerate(l2regs):
            mae, y_pred = predict_KRR(
                X_train, X_val, y_train, y_val, gamma=gamma, l2reg=l2reg, kernel=kernel
                )
           # print(f'mae={mae} for params {kernel, gamma, l2reg}')
            maes_lap[i, j] = mae
    min_i, min_j = np.unravel_index(np.argmin(maes_lap, axis=None), maes_lap.shape)
    min_gamma = gammas[min_i]
    min_l2reg_lap = l2regs[min_j]
    min_mae_lap = maes_lap[min_i, min_j]

    if min_sigma < min_gamma:
        print(f'best mae {min_mae_rbf} params rbf {min_sigma} {min_l2reg_rbf}')
        return 'rbf', min_sigma, min_l2reg_rbf
    else:
        print(f'best mae {min_mae_lap} params laplacian {min_gamma} {min_l2reg_lap}')
        return 'laplacian', min_gamma, min_l2reg_lap

def predict_CV(X, y, CV=5, mode='krr', seed=1, test_size=0.2, save_hypers=False, save_file=''):

    print("Learning mode", mode)

    maes = np.zeros((CV))
    kernels=[]
    sigmas = []
    l2regs = []
    for i in range(CV):
        print("CV iteration", i)
        seed += i
        X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, random_state=seed, test_size=test_size)
        if mode == 'krr':
            X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, shuffle=False, test_size=0.5)
            print('train size', len(X_train), 'val size', len(X_val), 'test size', len(X_test))
        elif mode == 'rf':
            X_test = X_test_val
            y_test = y_test_val
            print('train size', len(X_train), 'test size', len(X_test))
        else:
            return ValueError('invalid argument for train mode. should be rf or krr.')

        # hyperparam opt 
        if mode == 'krr':
            print("Optimising hypers...")
            kernel, sigma, l2reg = opt_hyperparams(X_train, X_val, y_train, y_val)
            kernels.append(kernel)
            sigmas.append(sigma)
            l2regs.append(l2reg)
            print("Making prediction with optimal params...")
            mae, _ = predict_KRR(X_train, X_test, 
                                y_train, y_test, 
                                sigma=sigma, l2reg=l2reg, 
                                gamma=sigma, kernel=kernel)
        elif mode == 'rf':
            mae, _ = predict_RF(X_train, X_test, y_train, y_test)
        else:
            return ValueError('invalid argument for train mode. should be rf or krr.')
        maes[i] = mae

    if save_hypers:
        print(f'saving hypers to {save_file}')
        hypers = {"CV iter":np.arange(CV), "kernel":kernels, "sigma/lambda":sigmas, "l2reg":l2regs}
        df = pd.DataFrame(hypers)
        df.to_csv(save_file)
    return maes

def learning_curve_KRR(X, y, CV=5, n_points=5, seed=1, test_size=0.2, save_hypers=False, save_file=''):
    tr_fractions = np.logspace(-1, 0, num=n_points, endpoint=True)
    maes = np.zeros((CV, n_points))
    kernels = []
    sigmas = []
    l2regs =[]

    for i in range(CV):
        print("CV iteration",i)
        seed += i
        X_train_all, X_test_val, y_train_all, y_test_val = train_test_split(X, y, random_state=seed, test_size=test_size)
        X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, shuffle=False, test_size=0.5)

        print("Optimising hypers...")
        kernel, sigma, l2reg = opt_hyperparams(X_train_all, X_val, y_train_all, y_val)
        kernels.append(kernel)
        sigmas.append(sigma)
        l2regs.append(l2reg)

        tr_sizes = [int(tr_fraction * len(X_train_all)) for tr_fraction in tr_fractions]
        print('tr sizes', tr_sizes)
        for j, tr_size in enumerate(tr_sizes):
            X_train = X_train_all[:tr_size]
            y_train = y_train_all[:tr_size]
            print(f"dataset size {len(X)}, train size {len(X_train)}, test size {len(X_test)}, val size {len(X_val)}")
            mae, _ = predict_KRR(X_train, X_test, y_train, y_test, sigma=sigma, l2reg=l2reg, gamma=sigma, kernel=kernel)
            maes[i,j] = mae

    if save_hypers:
        print(f'saving hypers to {save_file}')
        hypers = {"CV iter":np.arange(CV), "kernel":kernels, "sigma/lambda":sigmas, "l2reg":l2regs}
        df = pd.DataFrame(hypers)
        df.to_csv(save_file)

    return tr_sizes, maes


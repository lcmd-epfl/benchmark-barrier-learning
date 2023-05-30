import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from sklearn.model_selection import train_test_split, KFold
from scipy.spatial import distance_matrix
import xgboost as xgb

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
    sigmas=[0.1, 1, 10, 100, 1000],
    gammas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
    l2regs=[1e-10, 1e-7, 1e-4],
    kernel='rbf'
):
    """Optimize hyperparameters for KRR with gaussian or 
    laplacian kernel.
    """
    print(f"Hyperparam opt with {kernel} kernel")

    if kernel == 'rbf':
        sigmas = sigmas
    elif kernel == 'laplacian':
        sigmas = gammas
    maes = np.zeros((len(sigmas), len(l2regs)))

    for i, sigma in enumerate(sigmas):
        for j, l2reg in enumerate(l2regs):
            if kernel == 'rbf':
                mae, y_pred = predict_KRR(
                    X_train, X_val, y_train, y_val, sigma=sigma, l2reg=l2reg, kernel=kernel
                )
                print(f"MAE {mae} for sigma {sigma} and l2reg {l2reg}")
            if kernel == 'laplacian':
                mae, y_pred = predict_KRR(
                    X_train, X_val, y_train, y_val, gamma=sigma, l2reg=l2reg, kernel=kernel
                )
                print(f"MAE {mae} for gamma {sigma} and l2reg {l2reg}")
            maes[i, j] = mae

    min_j, min_k = np.unravel_index(np.argmin(maes, axis=None), maes.shape)
    min_sigma = sigmas[min_j]
    min_l2reg = l2regs[min_k]

    print(
        "min mae",
        mean_maes[min_j, min_k],
        "for sigma/gamma=",
        min_sigma,
        "and l2reg=",
        min_l2reg,
    )
    return min_sigma, min_l2reg

def predict_CV(X, y, CV=5, kernel='rbf', mode='krr', seed=1, test_size=0.2):

    print("Learning mode", mode)

    maes = np.zeros((CV))

    for i in range(CV):
        print("CV iteration", i)
        seed += i
        X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, random_state=seed, test_size=test_size)
        X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, shuffle=False, test_size=0.5)
        print('train size', len(X_train), 'val size', len(X_val), 'test size', len(X_test))

        # hyperparam opt 
        if mode == 'krr':
            print("Optimising hypers...")
            sigma, l2reg = opt_hyperparams(X_train, X_val, y_train, y_val, kernel=kernel)

            mae, _ = predict_KRR(X_train, X_test, 
                                y_train, y_test, 
                                sigma=sigma, l2reg=l2reg, 
                                gamma=sigma, kernel=kernel)
        elif mode == 'rf':
            mae, _ = predict_RF(X_train, X_test, y_train, y_test)
        else:
            print('mode is not recognised. exiting')
            return 
        maes[i] = mae

    return maes



from Block_SSC import Block_SSC
from LRR_model import demo

from scipy.linalg import fractional_matrix_power
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from scipy import linalg
import statsmodels.api as sm
import find_hyperparameter as findy


np.random.random(6)
lam = [0.5, 1, 5, 10, 20, 60, 100, 200];
gam = [0.8, 1, 5, 10, 20, 60, 100, 200];


def find_lrr(data):
    [B, E] = demo(np.array(data.values.tolist()))

    W = np.maximum(0, (B + B.T) / 2)
    W = W - np.diag(np.diag(B))
    D = np.diag(np.sum(W, axis=1))
    I = np.eye(W.shape[1])
    # L = D-B
    D_half = D ** (-0.5)
    whereinf = np.isinf(D_half)
    D_half[whereinf] = 0
    L_norm = I - D_half @ W @ D_half
    # if np.isnan(L_norm).any() == True:
    #     L_norm = D - W

    eigenvals, eigenvcts = linalg.eig(L_norm)
    eigenvals = np.real(eigenvals)
    eigenvals_sorted_indices = np.argsort(eigenvals)
    eigenvals_sorted = eigenvals[eigenvals_sorted_indices]
    i = 0
    while i < len(eigenvals_sorted) - 2:
        diff1 = np.abs(math.exp(eigenvals_sorted[i + 1] - eigenvals_sorted[i]))
        diff2 = np.abs(math.exp(eigenvals_sorted[i + 2] - eigenvals_sorted[i + 1]))
        diff = np.abs(diff1- diff2)
        i += 1
        if diff > 1.63:
            break

    return [B, i + 1]


def find_KBDR(data, win_size, lam, gam):
    #[B, regime_number] = find_lrr(data)
    regime_number =1
    [B] = findy.find_ssc(data)
    #print('regime_number=' + str(regime_number))
    B = np.maximum(0, (B + B.T) / 2)
    B = B - np.diag(np.diag(B))
    degree = np.diag(np.sum(B, axis=1))
    D_half = degree ** (-0.5)
    whereinf = np.isinf(D_half)
    D_half[whereinf] = 0

    B = D_half @ B @ D_half

    B = B + np.diag(np.sum(B, axis=1)) + 0.00001 * np.eye(B.shape[0])
    [Z, Z1] = Block_SSC(B, B.shape[1],
                        win_size, lam[5],
                        gam[0]).KBDR_solver(K=4)
    return [Z, Z1, regime_number, D_half]


def find_BDR(data, win_size, lam, gam):
    #[B, regime_number] = find_lrr(data)
    #print('regime_number=' + str(regime_number))
    print('begin BDR')
    [Z, Z1] = Block_SSC(np.array(data.values.tolist()), data.shape[1],
                        win_size, lam[5],
                        gam[0]).BDR_solver(K=4)
    return [Z, Z1]


def Multi_matrix(data, win_size, win_number, lam, gam):
    Z = []
    for i in range(win_number):
        [globals()['Z%s' % (i + 1)], Z_notuse] = find_KBDR(data.iloc[i * win_size:(i + 1) * win_size, :], win_size, lam,
                                                           gam)
        Z.append(globals()['Z%s' % (i + 1)])
    return Z


def KSRM(data, indices, win_size):
    Z = []
    regime_num = []
    D = []
    for i in tqdm(range(len(indices))):
        [globals()['Z%s' % (i + 1)], Z_notuse, number, Degree] = find_KBDR(pd.DataFrame(data[i]), win_size, lam,
                                                                           gam)
        if np.all(globals()['Z%s' % (i + 1)] == 0):
            Z.append(Z_notuse)
        else:
            Z.append(globals()['Z%s' % (i + 1)])

        regime_num.append(number)
        D.append(Degree)

    return [Z, regime_num, D]


def KSRM2(data, indices, win_size):
    Z = []
    regime_num = []
    D = []
    # if parallel:
    #     executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
    #     tasks = (delayed(find_KBDR)(pd.DataFrame(data[i]), win_size, lam, gam) for i in tqdm(range(len(indices))))
    #     [Z, Z_notuse, number] = executor(tasks)
    for i in tqdm(range(len(indices))):
        [globals()['Z%s' % (i + 1)], Z1, number] = find_BDR(pd.DataFrame(data[i]), win_size, lam, gam)

        Z.append(globals()['Z%s' % (i + 1)])

        regime_num.append(number)

    return [Z, regime_num]


def mar(X, pred_step, maxiter=100):
    T, m, n = X.shape
    B = np.random.randn(n, n)
    for it in range(maxiter):
        temp0 = B.T @ B
        temp1 = np.zeros((m, m))
        temp2 = np.zeros((m, m))
        for t in range(1, T):
            temp1 += X[t, :, :] @ B @ X[t - 1, :, :].T
            temp2 += X[t - 1, :, :] @ temp0 @ X[t - 1, :, :].T
        try:
            A = temp1 @ np.linalg.inv(temp2)
        except:
            A = temp1 @ np.linalg.pinv(temp2)
        else:
            A = A

        temp0 = A.T @ A
        temp1 = np.zeros((n, n))
        temp2 = np.zeros((n, n))
        for t in range(1, T):
            temp1 += X[t, :, :].T @ A @ X[t - 1, :, :]
            temp2 += X[t - 1, :, :].T @ temp0 @ X[t - 1, :, :]
        # B = temp1 @ np.linalg.inv(temp2)
        try:
            B = temp1 @ np.linalg.inv(temp2)
        except:
            B = temp1 @ np.linalg.pinv(temp2)
        else:
            B = B
    tensor = np.append(X, np.zeros((pred_step, m, n)), axis=0)
    for s in range(pred_step):
        tensor[T + s, :, :] = A @ tensor[T + s - 1, :, :] @ B.T
    return tensor[- pred_step:, :, :]


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def regression_predict(Z, split_series, window_size):
    x_train = [0 for x in range(0, np.shape(Z)[1] * np.shape(Z)[1])]
    x_predict = [0 for x in range(0, np.shape(Z)[1] * np.shape(Z)[1])]
    data_series = [0 for x in range(0, window_size * np.shape(Z)[1])]
    # D_pre = [0 for x in range(0, np.shape(D)[1] * np.shape(D)[1])]
    for j in range(np.shape(Z)[0] - 5, np.shape(Z)[0] - 2):
        x_train = np.column_stack((x_train, np.array(Z[j]).reshape(-1, 1)))
        x_predict = np.column_stack((x_predict, np.array(Z[j + 1]).reshape(-1, 1)))
        data_series = np.column_stack((data_series, np.array(split_series[j + 1]).reshape(-1, 1)))
    x_train = pd.DataFrame(x_train)
    x_predict = pd.DataFrame(x_predict)
    data_series = pd.DataFrame(data_series)

    x_train = x_train.drop(x_train.iloc[:, 0], axis=1).T.reset_index(drop=True).T
    x_predict = x_predict.drop(x_predict.iloc[:, 0], axis=1).T.reset_index(drop=True).T
    data_series = data_series.drop(data_series.iloc[:, 0], axis=1).T.reset_index(drop=True).T
    y = pd.DataFrame(np.array(Z[np.shape(Z)[0] - 2]).reshape(-1, 1))

    #KNNR
    '''
    data=[]
    for i in range(np.shape(Z)[0]):
        data.append(split_series[i])

    x_train, y_train = split_sequences(np.array(Z)[0:-1],3)
    data_train, ydata_train = split_sequences(np.array(data)[0:-1],3)

    x_train = x_train.reshape(np.shape(x_train)[0], np.shape(x_train)[1]*np.shape(x_train)[2]*np.shape(x_train)[3])
    y_train = y_train.reshape(np.shape(y_train)[0], np.shape(y_train)[1]*np.shape(y_train)[2])

    data_train = data_train.reshape(np.shape(data_train)[0], np.shape(data_train)[1]*np.shape(data_train)[2]*np.shape(data_train)[3])
    ydata_train = ydata_train.reshape(np.shape(ydata_train)[0], np.shape(ydata_train)[1]*np.shape(ydata_train)[2])

    knn = KNeighborsRegressor(n_neighbors=2)
    knn.fit(x_train,y_train)
    Z_p = knn.predict(np.array(Z[-3:]).reshape(1,3*np.shape(Z[-3:])[1]*np.shape(Z[-3:])[2]))
    Z_predict = Z_p.reshape(np.shape(Z)[1],np.shape(Z)[2])

    knn1 = KNeighborsRegressor(n_neighbors=2)
    knn1.fit(data_train,ydata_train)
    data_p = knn1.predict(np.array(data[-3:]).reshape(1,3*np.shape(data[-3:])[1]*np.shape(data[-3:])[2]))
    X_predict = data_p.reshape(window_size, np.shape(Z)[1])
    '''



    # model = AutoReg(x_train, lags=[3])
    # model = VAR(np.array(x_train))
    model = sm.OLS(y, x_train)
    result = model.fit()
    # result.summary()

    # matrix autoregression
    '''
    x_predict = np.array(Z[0: np.shape(Z)[0] - 1])
    data_series = np.array(split_series[0: np.shape(Z)[0] - 1])
    Z_predict = mar(x_predict, 1)
    X_predict = mar(data_series, 1)
    Z_predict = result.predict(np.array(x_predict),steps=len(x_predict))
    X_predict = result.predict(np.array(data_series),steps=len(data_series))
    '''

    Z_predict = result.predict(np.array(x_predict))
    X_predict = result.predict(np.array(data_series))
    #D_predict = result.predict(D_pre)
    Z_predict = Z_predict.reshape(np.shape(Z)[1], np.shape(Z)[1])
    X_predict = X_predict.reshape(window_size, np.shape(Z)[1])
    # D_predict = D_predict.values.reshape(np.shape(D)[1], np.shape(D)[1])
    # D_predict = np.linalg.inv(D_predict)
    Forecast_matrix = X_predict @ Z_predict
    # Forecast_matrix = X_predict @ (D_predict @ Z_predict @ D_predict)

    return [Forecast_matrix, Z_predict]

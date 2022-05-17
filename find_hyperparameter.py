#import generate_order_data_one_windows as genord
from Block_SSC import Block_SSC
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import SpectralClustering
from SSC_model import ssc_model
import time
#import cvxpy as cp
#from cvxpy.atoms.elementwise.power import power
from LRR_model import demo
import threading
#import Error
from scipy.linalg import fractional_matrix_power
import pandas as pd


# gnt = gnt.T.sample(frac=1).reset_index(drop=True).T


# find lam
# def find_lam():
#     fig, ((ax0, ax1, ax2, ax3), (ax4, ax5, ax6, ax7)) = plt.subplots(2, 4, figsize=(10, 10))
#     for i in range(len(genord.lam)):
#         [globals()['B_test%s' % i], globals()['Z_test%s' % i]] = Block_SSC(np.array(gnt.values.tolist()), gnt.shape[1],
#                                                                            genord.win_size, genord.lam[i],
#                                                                            genord.gam[0]).BDR_solver()
#         sns.heatmap(globals()['B_test%s' % i], ax=locals()['ax%s' % i], annot=False, xticklabels=False,
#                     yticklabels=False, square=True, cmap="YlGnBu_r", cbar=True, cbar_kws={"shrink": .39})
#         locals()['ax%s' % i].set_title('lam=' + str(genord.lam[i]))
#     plt.show()
#     return


# find gam
# def find_gam():
#     fig, ((ax0, ax1, ax2, ax3), (ax4, ax5, ax6, ax7)) = plt.subplots(2, 4, figsize=(10, 10))
#     for i in range(len(genord.gam)):
#         [globals()['B_test%s' % i], globals()['Z_test%s' % i]] = Block_SSC(np.array(gnt.values.tolist()), gnt.shape[1],
#                                                                            genord.win_size, genord.lam[5],
#                                                                            genord.gam[i]).BDR_solver()
#         sns.heatmap(globals()['B_test%s' % i], ax=locals()['ax%s' % i], annot=False, xticklabels=False,
#                     yticklabels=False, square=True, cmap="YlGnBu_r", cbar=True, cbar_kws={"shrink": .39})
#         locals()['ax%s' % i].set_title('gam=' + str(genord.gam[i]))
#     plt.show()
#     return


# def find_OSC_lam1():
#     fig, ((ax0, ax1, ax2, ax3), (ax4, ax5, ax6, ax7)) = plt.subplots(2, 4, figsize=(10, 10))
#     for i in range(len(genord.lambda_1)):
#         [globals()['Z%s' % i], funVal] = Block_SSC(np.array(gnt.values.tolist()), gnt.shape[1],
#                                                    genord.win_size, genord.lam[5],
#                                                    genord.gam[0]).OSC(genord.lambda_1[i], genord.lambda_2,
#                                                                       genord.gamma_1, genord.gamma_2, genord.p,
#                                                                       genord.maxIteration)
#         sns.heatmap(globals()['Z%s' % i], vmin=0, ax=locals()['ax%s' % i], annot=False, xticklabels=False,
#                     yticklabels=False,
#                     square=True, cmap="YlGnBu_r", cbar=True, cbar_kws={"shrink": .39})
#         locals()['ax%s' % i].set_title('lambda_1=' + str(genord.lambda_1[i]))
#         plt.show()
#     return


# def find_Block(data, K):
#     [B, Z] = Block_SSC(np.array(data.values.tolist()), data.shape[1],
#                        genord.win_size, genord.lam[5],
#                        genord.gam[0]).BDR_solver(K)
#     '''sns.heatmap(B, annot=False, xticklabels=False, yticklabels=False, square=True, cmap="YlGnBu_r", cbar=True)
#     plt.title('Block')
#     plt.show()
# '''
#     return [B]


def find_ssc(data):
    B = ssc_model(data).computeCmat()
    B = np.maximum(0, (B + B.T) / 2)
    return [B]


def find_sparse_sol(Y, i, N, D):
    if i == 0:
        Ybari = Y[:, 1:N]
    if i == N - 1:
        Ybari = Y[:, 0:N - 1]
    if i != 0 and i != N - 1:
        Ybari = np.concatenate((Y[:, 0:i], Y[:, i + 1:N]), axis=1)
    yi = Y[:, i].reshape(D, 1)

    # this ci will contain the solution of the l1 optimisation problem:
    # min (||yi - Ybari*ci||F)^2 + lambda*||ci||1   st. sum(ci) = 1

    ci = cp.Variable(shape=(N - 1, 1))
    constraint = [cp.sum(ci) == 1]
    obj = cp.Minimize(power(cp.norm(yi - Ybari @ ci, 2), 2) + 0.082 * cp.norm(ci, 1))  # lambda = 0.082
    prob = cp.Problem(obj, constraint)
    prob.solve()
    return ci.value


def find_ssc_cvxpy(input_data):
    input_data = np.array(input_data.values.tolist())
    C = np.concatenate((np.zeros((1, 1)), find_sparse_sol(input_data, 0, input_data.shape[1], input_data.shape[0])),
                       axis=0)
    B = np.add(np.absolute(C), np.absolute(C.T))
    return B


def find_lrr(data):
    [B, E] = demo(np.array(data.values.tolist()))
    return [B, E]



def zij(Y, i, j, lam, N):
    if i == j:
        return 0.0
    else:
        numerator = np.exp(-(np.square(np.linalg.norm(Y[:, i] - Y[:, j], 2))) / lam)
        # print(numerator)
        # sum_i=0
        # sum_j=0
        # for h in range(N):
        #     if h!=i:
        #         sum_i += np.exp(-(np.square(np.linalg.norm(Y[:,i]-Y[:,h],2)))/lam)
        # for h in range(N):
        #     if h!=j:
        #         sum_j += np.exp(-(np.square(np.linalg.norm(Y[:,j]-Y[:,h],2)))/lam)
        return numerator


def find_ssce():
    N = gnt.shape[1]
    Z = np.zeros((N, N), dtype='float64')
    for i in range(N):
        for j in range(i + 1):
            Z[i, j] = Z[j, i] = zij(np.array(gnt.values.tolist()), i, j, 0.1957, N)
    return (Z + Z.T) / 2


# OSC
def find_osc():
    [Z, funVal] = Block_SSC(np.array(gnt.values.tolist()), gnt.shape[1],
                            genord.win_size, genord.lam[5],
                            genord.gam[0]).OSC(genord.lambda_1[3], genord.lambda_2, genord.gamma_1, genord.gamma_2,
                                               genord.p, genord.maxIteration)
    sns.heatmap(Z, vmin=0, annot=False, xticklabels=False, yticklabels=False, square=True, cmap="YlGnBu_r",
                cbar=True)
    plt.title('OSC')
    plt.show()
    return


if __name__ == '__main__':
    # t = threading.Thread(target=your_func)
    # t.setDaemon(True)
    # t.start()
    np.random.seed(6)  # 6
    [gnt, labelnotuse] = genord.generate_order()

    #LRR_Matrix
    #gnt = pd.read_csv('data/LRR_1518_noise6_B.csv', header=None, index_col=None)
    #SSCE Matrix

    #gnt = pd.read_csv('data/SSCE_1518_noise6_B.csv', header=None, index_col=None)
    Label = pd.read_csv('data/Label_gnt1518.csv', header=None, index_col=None)
    Label = np.array(Label.values.tolist())
    label_all_subjs = Label
    label_all_subjs = label_all_subjs - label_all_subjs.min() + 1
    label_all_subjs = np.squeeze(label_all_subjs)

    start = time.time()
    scores = []
    scores2 = []

    # gnt = np.maximum(0, (gnt + gnt.T) / 2)
    # gnt = gnt - np.diag(np.diag(gnt))
    # degree = np.diag(np.sum(gnt, axis=1))
    # gnt = fractional_matrix_power(degree, -0.5) @ gnt @ fractional_matrix_power(degree, -0.5)
    # gnt = gnt + np.diag(np.sum(gnt, axis=1)) + 0.001*np.eye(gnt.shape[0])
    # SSCE
    # B=find_ssce()
    # KBDR
    # [B, Z] = Block_SSC(np.array(gnt.values.tolist()), gnt.shape[1],
    #                    genord.win_size, genord.lam[5],
    #                    genord.gam[0]).KBDR_solver(K=5)
    # BDR
    # [B, Z] = Block_SSC(np.array(gnt.values.tolist()), gnt.shape[1],
    #                    genord.win_size, genord.lam[5],
    #                    genord.gam[0]).BDR_solver(K=5)

    # LRR
    B, E = find_lrr(gnt)
    B1 = np.maximum(0, (B + B.T) / 2)
    y_pred = SpectralClustering(n_clusters=5, random_state=0, affinity='precomputed').fit_predict(B1)
    misrate_x = Error.err_rate(label_all_subjs, y_pred)
    acc = 1 - misrate_x
    print("acc: %.4f" % acc)

    # for k in range(2, 10):
    #     y_pred = SpectralClustering(n_clusters=k, random_state=0, affinity='precomputed').fit_predict(B1)
    #     scores.append(metrics.silhouette_score(B, y_pred, metric='euclidean'))
    #     scores2.append(metrics.calinski_harabasz_score(B, y_pred))

    i = range(2, 10)
    # plt.plot(i, scores, 'g.-', i, np.log(scores2), 'b.-')
    # plt.sh
    print('time=', time.time() - start)

# sns.heatmap(B, annot=False, xticklabels=False, yticklabels=False, square=True, cmap="YlGnBu_r", cbar=True)
# plt.title('Block')
# plt.show()

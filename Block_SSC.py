import numpy as np
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.linalg import solve_sylvester


class Block_SSC:

    def __init__(self, data=None, n_series=None, dim=None, lam=None, gam=None):
        self.data = data
        self.n_series = n_series
        self.dim = dim
        self.lam = lam
        self.gam = gam

    def BDR_solver(self, K):



        X = self.data
        n = X.shape[1]
        tol = 1e-3
        maxIter = 100
        one = np.ones((n, 1))
        XtX = X.T @ X #BDR
        #XtX = X #kernel BDR
        I = np.eye(n)
        invXtXI = np.linalg.inv(XtX + self.lam * I)
        gammaoverlambda = self.gam / self.lam
        Z = np.zeros((n, n))
        W = Z
        B = Z
        iter = 0

        while iter < maxIter:
            print(iter)
            iter = iter + 1
            # update Z
            Zk = Z
            Z = (invXtXI) @ (XtX + self.lam * B)

            # update B
            Bk = B
            B = Z - gammaoverlambda * (((np.diag(W)).reshape(n, 1)) @ (one.T) - W)
            B = np.maximum(0, (B + B.T) / 2)
            B = B - np.diag(np.diag(B))

            L = np.diag((B @ one).reshape(n)) - B


            # update W

            eigenvals, eigenvcts = linalg.eig(L)
            eigenvals = np.real(eigenvals)
            eigenvcts = np.real(eigenvcts)
            D = eigenvals.reshape(n, 1)
            eigenvals_sorted_indices = np.argsort(eigenvals)

            indices = []
            for i in range(0, K):
                ind = [];
                ind.append(eigenvals_sorted_indices[i]);
                indices.append(np.asarray(ind))
            indices = np.asarray(indices)
            zero_eigenvals_index = np.array(indices)
            W = (eigenvcts[:, eigenvals_sorted_indices[0:K]]) @ (eigenvcts[:, eigenvals_sorted_indices[0:K]].T);

            # stop ending
            diffZ = np.max(np.absolute(Z - Zk));
            diffB = np.max(np.absolute(B - Bk));
            stopC = np.maximum(diffZ, diffB);
            print(stopC)
            if stopC < tol:
                break;
        return B, Z
    def KBDR_solver(self, K):



        X = self.data
        n = X.shape[1]
        tol = 1e-3
        maxIter = 100
        one = np.ones((n, 1))
        #XtX = X.T @ X #BDR
        XtX = X #kernel BDR
        I = np.eye(n)
        invXtXI = np.linalg.inv(XtX + self.lam * I)
        gammaoverlambda = self.gam / self.lam
        Z = np.zeros((n, n))
        W = Z
        B = Z
        iter = 0

        while iter < maxIter:
            print(iter)
            iter = iter + 1
            # update Z
            Zk = Z
            Z = (invXtXI) @ (XtX + self.lam * B)

            # update B
            Bk = B
            B = Z - gammaoverlambda * (((np.diag(W)).reshape(n, 1)) @ (one.T) - W)
            B = np.maximum(0, (B + B.T) / 2)
            B = B - np.diag(np.diag(B))

            L = np.diag((B @ one).reshape(n)) - B


            # update W

            eigenvals, eigenvcts = linalg.eig(L)
            eigenvals = np.real(eigenvals)
            eigenvcts = np.real(eigenvcts)
            D = eigenvals.reshape(n, 1)
            eigenvals_sorted_indices = np.argsort(eigenvals)

            indices = []
            for i in range(0, K):
                ind = [];
                ind.append(eigenvals_sorted_indices[i]);
                indices.append(np.asarray(ind))
            indices = np.asarray(indices)
            zero_eigenvals_index = np.array(indices)
            W = (eigenvcts[:, eigenvals_sorted_indices[0:K]]) @ (eigenvcts[:, eigenvals_sorted_indices[0:K]].T);

            # stop ending
            diffZ = np.max(np.absolute(Z - Zk));
            diffB = np.max(np.absolute(B - Bk));
            stopC = np.maximum(diffZ, diffB);
            # print(stopC)
            if stopC < tol:
                break
        return B, Z
    def OSC(self, lambda_1, lambda_2, gamma_1, gamma_2, p, maxIteration):
        X = self.data
        funVal = np.zeros((maxIteration, 1))
        xn = X.shape[1]
        S = np.zeros((xn, xn))
        R = (np.triu(np.ones((xn, xn - 1)), 1) - np.triu(np.ones((xn, xn - 1)))) + (
                np.triu(np.ones((xn, xn - 1)), -1) - np.triu(np.ones((xn, xn - 1))))
        U = np.zeros((xn, xn - 1))
        G = np.zeros((xn, xn))
        F = np.zeros((xn, xn - 1))

        kron_Xt_X = sparse.csr_matrix(sparse.kron(np.eye(xn, xn), (X.T @ X)))
        kron_R_Rt = sparse.csr_matrix(sparse.kron(R @ R.T, np.eye(xn, xn)))

        iter = 0
        while iter < maxIteration:
            iter = iter + 1
            # update Z
            V = S - (G / gamma_1)
            [vm, vn] = V.shape
            rolled_v = V.reshape(vm * vn, 1)
            rolled_z = self.shrink_l1(rolled_v, lambda_1 / gamma_1)
            Z = rolled_z.reshape(vm, vn)
            Z = np.maximum(0, (Z + Z.T) / 2)

            Z = Z - np.diag(np.diag(Z))

            # update S
            # left=kron_Xt_X+sparse.csr_matrix(sparse.kron(np.eye(xn,xn),gamma_1*np.eye(xn,xn)))+gamma_2*kron_R_Rt;
            # right=X.T@X+gamma_2*U@R.T+gamma_1*Z+G+F@R.T;
            # right=right.reshape(xn*xn,1);
            # s=sparse.linalg.spsolve(left,right);
            # S=s.reshape(xn,xn);
            S = solve_sylvester(a=X.T @ X + gamma_1 * np.eye(xn, xn), b=gamma_2 * R @ R.T,
                                q=X.T @ X + gamma_2 * U @ R.T + gamma_1 * Z + G + F @ R.T)

            # update U
            V = S @ R - (1 / gamma_2) * F
            U = self.mysolve_l1l2(V, lambda_2 / gamma_2)

            # update G
            G = G + gamma_1 * (Z - S)

            # update F
            F = F + gamma_2 * (U - S @ R)

            # update gamma_1 and gamma_2
            gamma_1 = p * gamma_1
            gamma_2 = p * gamma_2

            funVal[iter] = 0.5 * (np.linalg.norm(X - X @ Z) ** 2) + lambda_1 * np.linalg.norm(Z,
                                                                                              ord=1) + lambda_2 * self.l2l1norm(
                Z @ R)
            print('the' + str(iter) + 'times iteration' + ',the funvalue is' + str(funVal[iter]))
            if iter > 1:
                if funVal[iter] < 1e-3:
                    break

            if iter > 100:
                if funVal[iter] < 1e-3 or funVal[iter - 1] - funVal[iter] < 1e-3:
                    break

        return Z, funVal

    def shrink_l1(self, b, c):
        x = np.zeros(np.size(b)).reshape(np.size(b), 1);
        k = np.where(b > c)
        x[k] = b[k] - c

        k = np.where(np.abs(b) <= c)
        x[k] = 0

        k = np.where(b < -c)
        x[k] = b[k] + c
        return x

    def mysolve_l2(self, w, c):
        nw = np.linalg.norm(w, ord=2)
        if nw > c:
            x = (nw - c) * w / nw
        else:
            x = np.zeros((np.size(w), 1))
        return x

    def mysolve_l1l2(self, W, c):
        n = W.shape[1]
        E = W
        for i in range(n):
            E[:, i] = self.mysolve_l2(W[:, i], c).reshape(W.shape[0])
        return E

    def l2l1norm(self, x):
        L = 0
        for i in range(x.shape[1]):
            L = L + np.linalg.norm(x[:, i], ord=2)

        return L

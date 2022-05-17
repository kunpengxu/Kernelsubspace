import pandas as pd
import numpy as np
from scipy.stats import zscore
import plotting as xplt

class Multi_division:

    def __init__(self, data=None, scale=None,name=None,root=None):
        self.data = data
        if len(self.data) == 780:
            self.scale = [len(data) // 3,1]
        else:
            self.scale = [len(data) // 3, 1]
        self.name= name
        self.root= root

    def smoothMAo(X, wd):
        wd = int(wd)
        if (wd == 1):
            return X
        # X[n][d]
        (n, d) = np.shape(X)
        Y = np.zeros((n, d))
        for i in range(0, d):
            Y[:, i] = np.convolve(X[:, i], np.ones(wd) / wd, mode='same')
        return Y

    def division(self):
        if len(self.data) == 780:
            data = self.data.apply(zscore)

        else:
            data = self.data.apply(zscore)

        XH = []
        X_tmp = data.to_numpy()
        HEIGHT = len(self.scale)
        for i in range(0, HEIGHT):
            h = self.scale[i]
            X_h = Multi_division.smoothMAo(X_tmp, h)
            XH.append(X_h)
            X_tmp = X_tmp - X_h
        XH = np.asarray(XH)
        return XH

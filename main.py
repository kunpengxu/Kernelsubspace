import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# sns.set()
import numpy as np
import Build_selfmatrix as bs

lam = [0.5, 1, 5, 10, 20, 60, 100, 200];
gam = [0.8, 1, 5, 10, 20, 60, 100, 200];

data = pd.read_csv('datasets/X_80d.csv', header=None)
data1 = data.iloc[:, 0:80].T

#[Z, Z1, regime_number, D_half] = bs.find_KBDR(data1, 80, lam, gam)
[Z,num]=bs.find_lrr(data1)
# sns.heatmap(Z, annot=False, xticklabels=False, yticklabels=False,
#               square=True, cmap="YlGnBu_r", cbar=False, cbar_kws={"shrink": .39})
sns.heatmap((Z > 0).astype(np.int_), annot=False, xticklabels=False, yticklabels=False,
            square=True, cmap="YlGnBu_r", cbar=False, cbar_kws={"shrink": .39})
plt.savefig('/Users/kunpengxu/PycharmProjects/Kernelsubspace/images/Syd_Z_bir.png', dpi=600,
            format='png',
            bbox_inches='tight')

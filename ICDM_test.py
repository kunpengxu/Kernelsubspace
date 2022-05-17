import pandas as pd
import matplotlib.pyplot as plt
X=pd.read_csv('datasets/X.csv', header=None)
X=X*20
plt.figure(figsize=(9, 4))
colorize = dict(c=X[2], cmap=plt.cm.get_cmap('rainbow', 5))
plt.scatter(X[0], X[1], **colorize)
#plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.show()
plt.savefig('/Users/kunpengxu/PycharmProjects/Kernelsubspace/images/ICDM.png', dpi=600,
            format='png',
            bbox_inches='tight')
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# [Pythonのコードのスター総数， Javaのコードのスターの総数, 年収]
X = np.array([[70, 30, 700],[32, 60, 480],[32, 20, 300],[20, 120, 600],[40, 120, 630], [40, 30, 520], [300, 1100, 1200], [2000, 400, 500],[40, 180, 800]])
pca = PCA(n_components=2)
pca.fit(X)

# 各主成分によってどの程度カバー出来ているかの割合
print(pca.explained_variance_ratio_)
# [ 0.80636224  0.17927921]

# 次元削減をXに適用する．
pca_point = pca.transform(X)

color = ['blue', 'green', 'red', 'yellow', 'brown', 'gray', 'deeppink', 'black','orange']
plt.scatter(*pca_point.T,  color=color)
plt.show()
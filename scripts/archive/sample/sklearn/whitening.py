import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# データの準備、確認
iris = datasets.load_iris()
nagasa = iris.data[:, 2]
haba = iris.data[:, 3]

plt.plot(nagasa[0:50], haba[0:50], 'o', label='setosa')
plt.plot(nagasa[50:100], haba[50:100], 'o', label='versicolor')
plt.plot(nagasa[100:150], haba[100:150], 'o', label='virginica')
plt.xlim([0, 8])
plt.ylim([-1, 3])
plt.xlabel('length')
plt.ylabel('breadth')
plt.title('Before Standardization')
plt.legend(loc='lower right')
plt.show()

# データの標準化
# 各変数を平均0、分散1にする
nagasa2 = nagasa - np.mean(nagasa)
haba2 = haba - np.mean(haba)
sd_nagasa = np.sqrt(sum(nagasa2*nagasa2)/np.size(nagasa2))
sd_haba = np.sqrt(sum(haba2*haba2)/np.size(haba2))

z_nagasa = nagasa2/sd_nagasa
z_haba = haba2/sd_haba

plt.plot(z_nagasa[0:50], z_haba[0:50], 'o', label='setosa')
plt.plot(z_nagasa[50:100], z_haba[50:100], 'o', label='versicolor')
plt.plot(z_nagasa[100:150], z_haba[100:150], 'o', label='virginica')
plt.xlim([-1.7, 2.0])
plt.ylim([-1.7, 2.0])
plt.xlabel('length')
plt.ylabel('breadth')
plt.title('After Standardization')
plt.legend(loc='lower right')
plt.show()


# 無相関化
# 固有ベクトル行列が回転行列になる．
covariance = 0.96921  # 正確には0.96921
Sigma = np.array([[sd_nagasa**2, covariance], [covariance, sd_haba**2]])
# Sigma = np.cov(z_nagasa, z_haba)とほぼ同じ
a, S = np.linalg.eig(Sigma)  # 固有値行列, 固有ベクトル行列

X = iris.data[:, 2:4]
Y = X.dot(S)

X = iris.data[:, 2:4]
Y = X.dot(S)

plt.plot(Y[0:50, 0], Y[0:50, 1], 'o', label='setosa')
plt.plot(Y[50:100, 0], Y[50:100, 1], 'o', label='versicolor')
plt.plot(Y[100:150, 0], Y[100:150, 1], 'o', label='virginica')
plt.xlim([0., 7.5])
plt.ylim([-2, 3.0])
plt.xlabel('length')
plt.ylabel('breadth')
plt.title('After Decorrelation')
plt.legend(loc='upper right')
plt.show()

# 白色化

Lambda = np.linalg.inv(S).dot(Sigma).dot(S)  # 分散共分散行列

xmu = X-np.mean(X, 0)
Lambda_invhalf = np.linalg.inv(np.sqrt(np.diag(Lambda)) * np.identity(2))
whiteX = xmu.dot(S).dot(Lambda_invhalf.transpose())  # 教科書の式4.17

plt.plot(whiteX[0:50, 0], whiteX[0:50, 1], 'o', label='setosa')
plt.plot(whiteX[50:100, 0], whiteX[50:100, 1], 'o', label='versicolor')
plt.plot(whiteX[100:150, 0], whiteX[100:150, 1], 'o', label='virginica')
plt.xlim([-3., 3.])
plt.ylim([-3., 3.])
plt.xlabel('length')
plt.ylabel('breadth')
plt.title('After Whitening')
plt.legend(loc='lower left')
plt.show()

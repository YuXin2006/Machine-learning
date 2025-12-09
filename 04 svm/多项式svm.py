import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
data = np.array([
    [0.1, 0.7],
    [0.3, 0.6],
    [0.4, 0.1],
    [0.5, 0.4],
    [0.8, 0.04],
    [0.42, 0.6],
    [0.9, 0.4],
    [0.6, 0.5],
    [0.7, 0.2],
    [0.7, 0.67],
    [0.27, 0.8],
    [0.5, 0.72]
])# 建立数据集
label = [1] * 6 + [0] * 6 #前六个数据的label为1后六个为0
x_min, x_max = data[:, 0].min() - 0.2, data[:, 0].max() + 0.2
y_min, y_max = data[:, 1].min() - 0.2, data[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.002),
                     np.arange(y_min, y_max, 0.002))

plt.figure(figsize=(16, 15))

for i, degree in enumerate([1, 3, 5, 7, 9, 12]):  # 多项式次数选择了1,3,5,7,9,12
    # C: 惩罚系数，gamma: 高斯核的系数
    model_poly = svm.SVC(C=0.0001, kernel='poly', degree=degree)  # 多项式核
    model_poly.fit(data, label)  # 训练
    # ravel - flatten
    # c_ - vstack
    # 把后面两个压扁之后变成了x1和x2，然后进行判断，得到结果在压缩成一个矩形
    Z = model_poly.predict(np.c_[xx.ravel(), yy.ravel()])  # 预测
    Z = Z.reshape(xx.shape)

    plt.subplot(3, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.6)

    # 画出训练点
    plt.scatter(data[:6, 0], data[:6, 1], marker='o', color='r', s=100, lw=3)
    plt.scatter(data[6:, 0], data[6:, 1], marker='x', color='k', s=100, lw=3)
    plt.title('Poly SVM with $\degree=$' + str(degree))
plt.show()
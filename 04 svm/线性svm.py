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
label = [1] * 6 + [0] * 6 #定义标签类别 前六个点为1 后六个点为0
#绘制网格图像的范围 x的范围和y的范围都在（min-0.2，max+0.2）
x_min, x_max = data[:, 0].min() - 0.2, data[:, 0].max() + 0.2
y_min, y_max = data[:, 1].min() - 0.2, data[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.002),
                    np.arange(y_min, y_max, 0.002)) # meshgrid如何生成网格（np.meshgrid 接受两个一维坐标数组，然后返回两个二维矩阵 xx 和 yy。这两个矩阵包含了绘图区域内所有点的 X 坐标和 Y 坐标。步长 0.002
print(xx)#打印矩阵


model_linear = svm.SVC(kernel='linear', C = 0.001)# 线性svm分类器 c=0.001是正则化参数 值小 边界更平滑 软间隔程度越大
model_linear.fit(data, label) # 训练，训练参数为特征值和类别
Z = model_linear.predict(np.c_[xx.ravel(), yy.ravel()]) # 预测
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap = plt.cm.ocean, alpha=0.6)
plt.scatter(data[:6, 0], data[:6, 1], marker='o', color='r', s=100, lw=3)
plt.scatter(data[6:, 0], data[6:, 1], marker='x', color='k', s=100, lw=3)
plt.title('Linear SVM')
plt.show()
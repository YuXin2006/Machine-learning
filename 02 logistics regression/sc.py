import numpy as np
from sklearn.datasets import fetch_openml#导入数据获取工具fetch_openml
from sklearn.linear_model import LogisticRegression


mnist=fetch_openml('mnist_784')#下载并加载mnist数据集
X, y = mnist.data, mnist.target#x是图片数据 y是答案
#训练集前60000张照片
X_train=np.array(X[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
#测试集后10000张照片
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(y[60000:], dtype=float)
#打印训练集矩阵形状
print(X_train.shape)
print(y_train.shape)
#打印测试集矩阵形状
print(X_test.shape)
print(y_test.shape)

clf = LogisticRegression(penalty="l1", solver="saga", tol=0.1)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)#得分为正确率
print("Test score with L1 penalty: %.4f" % score)

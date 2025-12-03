import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
def true_fun(X):
    return 1.5*X + 0.2

np.random.seed(0)
n_samples = 30

'''生成随机数据作为训练集，并且加一些噪声'''
X_train = np.sort(np.random.rand(n_samples))
y_train = (true_fun(X_train) + np.random.randn(n_samples) * 0.05).reshape(n_samples,1)

model = LinearRegression()
model.fit(X_train[:,np.newaxis], y_train)

print("输出参数w：",model.coef_) # 输出模型参数w
print("输出参数b：",model.intercept_) # 输出参数b

X_test = np.linspace(0, 1, 100)
plt.plot(X_test, model.predict(X_test[:, np.newaxis]), label="Model")
plt.plot(X_test, true_fun(X_test), label="True function")
plt.scatter(X_train,y_train) # 画出训练集的点
plt.legend(loc="best")
plt.show()
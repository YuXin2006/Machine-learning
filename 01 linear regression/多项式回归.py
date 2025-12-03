import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score #交叉验证法评估模型
from sklearn.preprocessing import PolynomialFeatures #生成多项式特征值
from sklearn.pipeline import Pipeline#把多项式特征值生成和线性回归拟合串联
def true_fun(X): # 生成一条真实的余弦曲线y = cos(1.5Πx)
    return np.cos(1.5 * np.pi * X)
np.random.seed(0)
n_samples = 30
#数据生成
X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1
#设置三种不同复杂的的模型
degrees = [1, 4, 15]
#拟合三种不同多项式级数的图形
for i in range(len(degrees)):
    #这是一个在 matplotlib 库中创建子图（一个小图表）的函数,接收三个整数参数：(行数, 列数, 当前子图位置)
    ax = plt.subplot(1, len(degrees), i + 1)
    # 这是一个设置（set）绘图对象属性（property）的通用函数
    plt.setp(ax, xticks=(), yticks=())#在子图上设置坐标轴刻度
    #特征转换 (PolynomialFeatures)：模型收到 X 后，首先根据 degrees[i] 的值，把它变成更复杂的特征组合。
    polynomial_features = PolynomialFeatures(degree=degrees[i],include_bias=False)
    #线性回归拟合
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)#命令模型开始拟合数据

    scores = cross_val_score(pipeline, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10)
    X_test = np.linspace(0, 1, 100)#x——test最终是一个包含 100 个数字的数组，它们从 0 开始，以微小的等间距递增，直到 1 结束。
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")#展示原始数据分布
    plt.xlabel("x")#设置x坐标轴
    plt.ylabel("y")#设置y坐标轴
    plt.xlim((0, 1))#设置x范围
    plt.ylim((-2, 2))#设置y范围
    plt.legend(loc="best")#在图上展示不同线条和点的名字，并让软件智能地选择一个最不影响你观察数据的位置。

    #为子图设置标题
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
          #用科学计数法格式化
          #mse：平均均方误差  std：标准差
plt.show()

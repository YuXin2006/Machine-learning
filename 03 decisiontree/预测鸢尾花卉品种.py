import seaborn as sns #高级统计图表库
from pandas import plotting#pandas绘图工具
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split#数据分割工具
from sklearn import tree
# 加载数据集
data = load_iris()
# 转换成.DataFrame形式
df = pd.DataFrame(data.data, columns = data.feature_names)
# 添加品种列
df['Species'] = data.target
# 查看数据集信息
print(f"数据集信息：\n{df.info()}")
# 查看前5条数据
print(f"前5条数据：\n{df.head()}")
# 查看各特征列的摘要信息
df.describe()


# 设置颜色主题
antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864']
# 绘制violinplot
f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
sns.despine(left=True) # 删除上方和右方坐标轴上不需要的边框
sns.violinplot(x='Species', y=df.columns[0], data=df, palette=antV, ax=axes[0, 0])
sns.violinplot(x='Species', y=df.columns[1], data=df, palette=antV, ax=axes[0, 1])
sns.violinplot(x='Species', y=df.columns[2], data=df, palette=antV, ax=axes[1, 0])
sns.violinplot(x='Species', y=df.columns[3], data=df, palette=antV, ax=axes[1, 1])
plt.show()
# 绘制pointplot
f, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
sns.despine(left=True)
sns.pointplot(x='Species', y=df.columns[0], data=df, color=antV[1], ax=axes[0, 0])
sns.pointplot(x='Species', y=df.columns[1], data=df, color=antV[1], ax=axes[0, 1])
sns.pointplot(x='Species', y=df.columns[2], data=df, color=antV[1], ax=axes[1, 0])
sns.pointplot(x='Species', y=df.columns[3], data=df, color=antV[1], ax=axes[1, 1])
plt.show()
# g = sns.pairplot(data=df, palette=antV, hue= 'Species')
# 安德鲁曲线
#高维投影： 这是 Pandas 绘图库中的一个高级函数 。
# 它将数据集的多维特征投影到二维平面上的傅里叶级数曲线上。如果相同物种的曲线聚集成一簇，而不同物种的曲线能清晰分开，说明这些特征组合在一起具有很好的区分性。
# 'Species' 是用于分组的列名。
plt.subplots(figsize = (8,6))
plotting.andrews_curves(df, 'Species', colormap='cool')

plt.show()

# 加载数据集
data = load_iris()
# 转换成.DataFrame形式
df = pd.DataFrame(data.data, columns = data.feature_names)
# 添加品种列
df['Species'] = data.target

# 用数值替代品种名作为标签
target = np.unique(data.target)
target_names = np.unique(data.target_names)
targets = dict(zip(target, target_names))
df['Species'] = df['Species'].replace(targets)

# 提取数据和标签
X = df.drop(columns="Species")
y = df["Species"]
feature_names = X.columns
labels = y.unique()

X_train, test_x, y_train, test_lab = train_test_split(X,y,
                                                 test_size = 0.4,
                                                 random_state = 42)
model = DecisionTreeClassifier(max_depth =3, random_state = 42)
model.fit(X_train, y_train)
# 以文字形式输出树
text_representation = tree.export_text(model)
print(text_representation)
# 用图片画出
plt.figure(figsize=(30,10), facecolor = 'white') #
a = tree.plot_tree(model,
                   feature_names = feature_names,
                   class_names = labels,
                   rounded = True,
                   filled = True,
                   fontsize=14)
plt.show()
from sklearn.metrics import classification_report

# 打印详细的分类性能报告
y_pred = model.predict(test_x)
report = classification_report(test_lab, y_pred)

print("\n--- 详细分类报告 ---")
print(report)
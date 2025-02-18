import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import matplotlib as mpt
import sys
import joblib

import warnings

warnings.filterwarnings("ignore")

# 加载数据
data = pd.read_csv('../../data/boston_housing.data', sep='\s+', header=None)

# 获取特征属性X和目标属性Y
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 特征工程
"""
PolynomialFeatures ####多项式扩展
degree=2,扩展的阶数
interaction_only=False,是否只保留交互项
include_bias=True，是否需要偏置项
"""
print(type(x_train))
print(x_train.shape)
print(x_test.shape)
print(x_test.iloc[0, :])
poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)

x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
print(type(x_train_poly))
print(x_train_poly.shape)
print(x_test_poly.shape)
print(x_test_poly[0])
joblib.dump(poly,"./poly.m")

# 构建模型
ridge  = Ridge(alpha=0.1, fit_intercept=True)

# 模型训练
ridge.fit(x_train_poly, y_train)
print("="*50)

print(ridge.coef_)
print(ridge.intercept_)

# 预测测试集
y_test_hat = ridge.predict(x_test_poly)
print("-" * 100)

print(ridge.score(x_train_poly, y_train))
print(ridge.score(x_test_poly, y_test))

y_train_hat = ridge.predict(x_train_poly)
plt.figure(num="train")
plt.plot(range(len(x_train)), y_train, 'r', label=u'true')
plt.plot(range(len(x_train)), y_train_hat, 'g', label=u'predict')
plt.legend(loc='upper right')
plt.title("train")
# plt.show()
plt.figure(num="test")
plt.plot(range(len(x_test)), y_test, 'r', label=u'true')
plt.plot(range(len(x_test)), y_test_hat, 'g', label=u'predict')
plt.legend(loc='upper right')
plt.title("test")
plt.show()
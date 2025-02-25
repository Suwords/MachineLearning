#%% md
# 使用 sklearn 实现KNN算法

#%%
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score

T = [
    [3, 104, -1],
    [2, 100, -1],
    [1, 81, -1],
    [101, 10, 1],
    [99, 5, 1],
    [98, 2, 1]]
##初始化待测样本
x = [[18, 90]]
##初始化邻居数
K = 3

data = pd.DataFrame(T, columns=['A', 'B', 'label'])
print(data)
X_train = data.iloc[:, :-1]
Y_train = data.iloc[:, -1]
print(X_train)
print(Y_train)

KNN01 = neighbors.KNeighborsClassifier(n_neighbors=3)
KNN01.fit(X_train, Y_train) # 训练模型
y_pred = KNN01.predict(x) # 预测数据
print(y_pred)

score = KNN01.score(X=X_train, y=Y_train)
print(score)

KNN02 = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
KNN02.fit(X_train, Y_train)
y_pred = KNN02.predict(x)
print(y_pred)
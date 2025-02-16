#%% md
# # KNN
# 
# 手动实现KNN算法，不调用 sklearn 的包
#%%
import numpy as np
import pandas as pd

# KNN 等权投票 分类
# 初始化数据，这里的数据将label进行了标量化，取代了原本的字符串
T = [[3, 104, -1],
     [2, 100, -1],
     [1, 81, -1],
     [101, 10, 1],
     [99, 5, 1],
     [98, 2, 1]]

# 初始化待测样本
x = [18, 90]

# 初始化邻居数 - 即KNN算法中的K值
K = 3

# 初始化存储距离列表[[距离1, 标签1], [距离2, 标签2]...]
listDistance = []

# 循环每一个数据点，把计算结果放入dis
for i in T:
    dis = np.sum((np.array(i[0:-1]) - np.array(x)) ** 2) ** 0.5
    listDistance.append([dis, i[-1]])

# 对dis按照距离排序
listDistance.sort()
print(listDistance)
# 将前K个票放入投票箱
arr = np.array(listDistance[:K])[:, -1]
a = pd.Series(arr).value_counts()
print(a.idxmax())
#%%
# KNN 加权投票 -- 分类
T = [
    [3, 104, -1],
    [2, 100, -1],
    [1, 81, -1],
    [101, 10, 1],
    [99, 5, 1],
    [98, 2, 1]]

x = [18, 90]
K = 3
listDistance = []
for i in T:
    dis = np.sum((np.array(i[0:-1]) - np.array(x)) ** 2) ** 0.5 # 欧式距离
    listDistance.append([dis, i[-1]])

listDistance.sort()
print(listDistance)
pre = -1 if sum([1 / i[0] * i[1] for i in listDistance[:K]]) < 0 else 1
print(pre)

#%%
# KNN等权 -- 回归
T = [
    [3, 104, 98],
    [2, 100, 93],
    [1, 81, 95],
    [101, 10, 16],
    [99, 5, 8],
    [98, 2, 7]]

x = [18, 90]
K = 5
listDistance = []
for i in T:
    dis = np.sum((np.array(i[0:-1]) - np.array(x)) ** 2) ** 0.5 # 欧式距离
    listDistance.append([dis, i[-1]])

listDistance.sort()
print(listDistance)
pre = np.mean(np.array(listDistance[:K])[:, -1])
print(pre)

#%%
# KNN加权回归
T = [
    [3, 104, 98],
    [2, 100, 93],
    [1, 81, 95],
    [101, 10, 16],
    [99, 5, 8],
    [98, 2, 7]]

x = [18, 90]
K = 5
listDistance = []
for i in T:
    dis = np.sum((np.array(i[0:-1]) - np.array(x)) ** 2) ** 0.5 # 欧式距离
    listDistance.append([dis, i[-1]])

listDistance.sort()
print(listDistance)
print([i for i in listDistance[:K]])
pre = np.sum([1 / i[0] * i[1] for i in listDistance[:K]]) / np.sum([1 / i[0] for i in listDistance[:K]])
print(pre)
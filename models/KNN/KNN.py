"""
简单实现等权分类 封装KNN类
"""

import numpy as np
import pandas as pd

from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class KNN:
    """
    KNN的步骤：
    1. 从训练集合中获取K个离待预测样本距离最近的样本数据
    2. 根据获取到的K个样本数据来预测当前待预测样本的目标属性值
    """

    def __init__(self, k, with_kd_tree=True):
        self.k = k
        self.with_kd_tree = with_kd_tree
        self.train_x = None
        self.train_y = None

    def fit(self, train_x, train_y):
        """
        fit 训练模型，实际就是保存训练数据
        :param train_x: 训练数据的特征矩阵
        :param train_y: 训练数据的label
        :return:
        """
        # 将数据转化为numpy数组的形式
        x = np.array(train_x)
        y = np.array(train_y)
        self.train_x = x
        self.train_y = y
        if self.with_kd_tree:
            self.kd_tree = KDTree(train_x, leaf_size=10, metric='minkowski')

    def fetch_k_neighbors(self, x):
        """
        1. 从训练集合中获取K个离待预测样本距离最近的样本数据
        2. 根据获取得到的K个样本数据来预测当前待预测样本的目标属性值
        :param x: 当前样本的特征属性x（一条样本）
        :return: K个最近邻样本的索引
        """
        if self.with_kd_tree:
            # 获取对应最近K个样本的标签
            index = self.kd_tree.query([x], k=self.k, return_distance=False)[0]
            print(index)
            k_neighbors_label = []
            for i in index:
                k_neighbors_label.append(self.train_y[i])
            print(k_neighbors_label)
            return k_neighbors_label
        else:
            # 定义一个列表用来存储每个样本的距离以及对应的标签
            listDistance = []
            for index, i in enumerate(self.train_x):
                dis = np.sum((np.array(i) - np.array(x)) ** 2) ** 0.5
                listDistance.append([dis, self.train_y[index]])
            # 按照dis对listDistance进行排序
            listDistance.sort()
            k_neighbors_label = np.array(listDistance)[:self.k, -1]
            return k_neighbors_label

    def predict(self, X):
        """
        模型预测
        :param x: 待预测样本的特征矩阵（多个样本）
        :return: 预测结果
        """

        X = np.asarray(X)

        result = []
        for x in X:
            k_neighbnors_label = self.fetch_k_neighbors(x)
            # 统计每个类别出现的次数
            y_count = pd.Series(k_neighbnors_label).value_counts()
            # 产生结果
            y_ = y_count.idxmax()
            result.append(int(y_))
        return result

    def score(self, x, y):
        """
        模型预测得分 使用准确率
        :param x:
        :param y:
        :return:
        """
        y_true = np.array(y)
        y_pred = self.predict(x)
        return accuracy_score(y_true, y_pred)

    def save_model(self, path):
        """
        保存模型
        :return:
        """

        pass

    def loada_model(self, path):
        """
        加载模型
        :param path:
        :return:
        """

if __name__ == '__main__':
    T = np.array([
        [3, 104, -1],
        [2, 100, -1],
        [1, 81, -1],
        [101, 10, 1],
        [99, 5, 1],
        [98, 2, 1]])
    X_train = T[:, :-1]
    Y_train = T[:, -1]

    x_test = [[18, 90], [50, 10]]
    knn = KNN(k=5, with_kd_tree=False)
    knn.fit(X_train, Y_train)
    # print(knn.predict(X_train))
    print(knn.score(X_train, Y_train))
    print('预测结果：{}'.format(knn.predict(x_test)))
    print('-----------下面测试一下鸢尾花数据-----------')
    from sklearn.datasets import load_iris

    X, Y = load_iris(return_X_y=True)
    print(X.shape, Y.shape)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(x_train.shape, y_train.shape)
    knn01 = KNN(3, with_kd_tree=False)
    knn01.fit(x_train, y_train)
    print(knn01.score(x_train, y_train))
    print(knn01.score(x_test, y_test))
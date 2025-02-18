import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import sys

# 设置字符集
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 构造数据
X1 = np.array([10, 15, 20, 30, 50, 60, 60, 70]).reshape((-1, 1))
Y = np.array([0.8, 1.0, 1.8, 2.0, 3.2, 3.0, 3.1, 3.5]).reshape((-1, 1))

# print(X1)
# print(Y)

# 添加一个截距项
X = np.column_stack((np.ones_like(X1), X1))

X = np.mat(X)
Y = np.mat(Y)

print(X)
print(Y)

# 求解
theta = (X.T * X).I * X.T * Y
print(theta)
# sys.exit()
# 根据求解出来的theta求出预测值
predict_y = X * theta
print(predict_y)
# 查看MSE和R^2
print(Y.shape)
print(predict_y.shape)
#mse = mean_squared_error(y_true=Y,y_pred=np.asarray(predict_y))
#print("MSE",mse)
#r2 = r2_score(y_true=Y,y_pred=predict_y)
#print("r^2",r2)

x_test = [[1, 55]]
y_test_hat = x_test * theta
print("价格：", y_test_hat)

# print(predict_y)
# 画图可视化
plt.plot(X1, Y, 'bo', label=u'真实值')
plt.plot(X1, predict_y, 'r--o', label=u'预测值')
plt.legend(loc='lower right')
plt.show()
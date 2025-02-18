import joblib
import KNN
## 1. 加载模型
knn = joblib.load("knn.m")

## 2. 对待预测数据进行预测
x = [[18, 90], [20, 40]]
y_hat = knn.predict(x)
print(y_hat)
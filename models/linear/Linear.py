import numpy as np

class Linear:
    def __init__(self, use_b=True):
        self.use_b = use_b
        self.theta = None
        self.theta0 = 0

    def train(self, X, Y):
        if self.use_b:
            X = np.column_stack((np.ones((X.shape[0], 1)), X))

        X = np.mat(X)
        Y = np.mat(Y)

        theta = (X.T * X).I * X.T * Y
        if self.use_b:
            self.theta0 = theta[0]
            self.theta = theta[1:]
        else:
            self.theta0 = 0
            self.theta = theta

    def predict(self, X):
        predict_y = X * self.theta + self.theta0
        return predict_y

    def score(self, X, Y):
        pass

    def save(self):
        pass

    def load(self, model_path):
        pass

if __name__ == "__main__":
    X1 = np.array([10, 15, 20, 30, 50, 60, 60, 70]).reshape((-1, 1))
    Y = np.array([0.8, 1.0, 1.8, 2.0, 3.2, 3.0, 3.1, 3.5]).reshape((-1, 1))
    linear = Linear(use_b=True)
    linear.train(X1, Y)
    x_test = [[55]]
    y_test_hat = linear.predict(x_test)
    print(y_test_hat)
    print(linear.theta)
    print(linear.theta0)
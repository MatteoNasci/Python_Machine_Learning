from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class MyTrumpiniLinearRegression:
    def __init__(self, max_iter=1000, learning_rate=0.00001):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        pass

    def fit(self, X, y):
        self.w = np.random.rand(len(X[0]))
        for epoch in range(self.max_iter):
            for j, x in enumerate(X):
                p = self.predict_single(x)
                error = p - y[j]
                for i, x_i in enumerate(x):
                    self.w[i] -= self.learning_rate * x_i * error  # DELTA RULE

    def predict(self, X):
        # usare reduce somma su x*w?
        return [self.predict_single(x) for x in X]

    def predict_single(self, x):
        accumulator = 0.0
        for i, x_i in enumerate(x):
            accumulator += x_i * self.w[i]
        return accumulator


if __name__ == "__main__":
    dataset = load_boston()
    X = dataset['data']
    y = dataset['target']

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    model = MyTrumpiniLinearRegression(max_iter=50_000,learning_rate=0.001)
    model.fit(X, y)
    predicted = model.predict(X)
    mae = mean_absolute_error(y, predicted)
    print("Mean absolute error: ")
    print(mae)

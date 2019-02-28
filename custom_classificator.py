from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.model_selection import train_test_split


class MyTrumpiniLogisticRegression:
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
        return np.round(1 / (1 + pow(np.e, -accumulator)))


if __name__ == "__main__":
    dataset = load_breast_cancer()
    X = dataset['data']
    y = dataset['target']

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = MyTrumpiniLogisticRegression(max_iter=1_000, learning_rate=0.001)
    model.fit(X_train, y_train)

    predicted = model.predict(X_train)
    mae = mean_absolute_error(y_train, predicted)
    r2 = r2_score(y_train, predicted)
    accuracy = accuracy_score(y_train, predicted)
    print("Train Mean absolute error: ")
    print(mae)
    print("Train r2 error: ")
    print(r2)
    print("Train accuracy: ")
    print(accuracy)
    predicted = model.predict(X_test)
    mae = mean_absolute_error(y_test, predicted)
    r2 = r2_score(y_test, predicted)
    accuracy = accuracy_score(y_test, predicted)
    print("Test Mean absolute error: ")
    print(mae)
    print("Test r2 error: ")
    print(r2)
    print("Test accuracy: ")
    print(accuracy)

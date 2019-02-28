from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# sebbene siano chiamati Regressors sono dei classificatori
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import custom_classificator

dataset = load_iris()
X = dataset['data']
y = dataset['target']

""" In questo caso specifico per qualche motivo lo scaling peggiora le prestazioni del modello finale
scaler = QuantileTransformer()
X = scaler.fit_transform(X)
"""

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression(max_iter=500_000)
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

train_accuracy = accuracy_score(y_train, p_train)
test_accuracy = accuracy_score(y_test, p_test)

print('Train accuracy: ')
print(train_accuracy)
print('Test accuracy: ')
print(test_accuracy)

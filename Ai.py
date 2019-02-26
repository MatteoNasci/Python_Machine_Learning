"""
https://scikit-learn.org/stable/index.html
Classification
Regression
Clustering
Dimensionality reduction
Model selection 
Preprocessing
"""
import numpy as numpy
from pprint import pprint as pp
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error


def distantiate_print():
    print()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print()
    return None


def distantiate_print_obj(obj):
    print(obj)
    return distantiate_print()


data_set = load_boston()
X = data_set['data']
y = data_set['target']
model = LinearRegression()
model.fit(X, y)
p = model.predict(X)

test_enumerate = ['ciao', 'due', 'bye']
for i, i_a in enumerate(test_enumerate):
    print(i, i_a, test_enumerate[i])

distantiate_print()
distantiate_print_obj(data_set)
distantiate_print_obj(data_set.keys())
distantiate_print_obj(data_set['DESCR'])
distantiate_print_obj(X)
distantiate_print_obj(X[0])
distantiate_print_obj(y)
distantiate_print_obj(y[0])
distantiate_print_obj(p)
distantiate_print_obj(model.coef_)

min_square_error = 0
min_square_error_reduced = 0
min_absolute_error = 0

for i, y_i in enumerate(y):
    original_target = y[i]
    predicted_target = p[i]
    error = original_target - predicted_target
    print('Target', original_target, 'Prediction',
          predicted_target, '\t\tError', error)
    min_square_error_reduced += (error**2)**0.5
    min_square_error += error**2
    min_absolute_error += abs(error)

print()

min_square_error /= len(y)
min_absolute_error /= len(y)
min_square_error_reduced /= len(y)

mae = y - p
mae /= len(y)

mse = (y - p)**2
mse /= len(y)

print('Min square error', min_square_error, 'Min square error reduced',
      min_square_error_reduced, 'Min absolute error', min_absolute_error)

mse_official = mean_squared_error(y, p)
mae_official = mean_absolute_error(y, p)

print('Official min square error', min_square_error,
      'Official min absolute error', min_absolute_error)

print('My min square error', mse_official,
      'My min absolute error', mae_official)

distantiate_print()

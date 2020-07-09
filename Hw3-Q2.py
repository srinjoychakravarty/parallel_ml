from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot
import numpy
import time
# load data
data = read_csv('train.csv')
dataset = data.values
# split data into X and y
X = dataset[:30000,0:94]
y = dataset[:30000,94]
# encode string class values as integers
label_encoded_y = LabelEncoder().fit_transform(y)
# grid search
model = XGBClassifier()
n_estimators = [100, 200, 300, 400, 500]
learning_rate = [0.0001, 0.001, 0.01, 0.1]
max_depth = [1]
subsample = [1.0]
num_threads = [8,6,4,2,1]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample= subsample)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
for n in num_threads:
    print(f'Number of Threads: {n}')
    start = time.time()
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=n, cv=kfold)
    grid_result = grid_search.fit(X, label_encoded_y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
      	print("%f (%f) with: %r" % (mean, stdev, param))
    # plot results
    scores = numpy.array(means).reshape(len(learning_rate), len(n_estimators))
    elapsed = time.time() - start
    print(f'Elapsed Duration: {elapsed}')

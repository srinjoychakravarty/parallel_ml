from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')  # prevents matplotlib error on discovery cluster
from matplotlib import pyplot
import numpy
import time

data = read_csv('train.csv')
dataset = data.values

X = dataset[:12000, 0:94]   # split data into X and y and sampling from total row entries to save time
y = dataset[:12000, 94]     

label_encoded_y = LabelEncoder().fit_transform(y) 

model = XGBClassifier()
number_of_trees = [100, 200, 300, 400, 500]
# number_of_trees = [60, 70, 80, 90, 100]
max_depth = [1]
subsample = [1.0]
colsample_bytree = [0.18]
learning_rate = [0.0001, 0.001, 0.01, 0.1]
gamma = [0] 
objective = ['binary:logistic']
reg_alpha = [0.2]
reg_lambda = [0.4]
num_threads = [8, 7, 6, 5, 4, 3, 2, 1]
param_grid = {'reg_lambda': reg_lambda,'reg_alpha': reg_alpha, 'objective': objective, 'gamma': gamma, 'colsample_bytree': colsample_bytree, 'subsample': subsample, 'max_depth': max_depth, 'n_estimators': number_of_trees, 'learning_rate': learning_rate}
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 93)

for n in num_threads:
    print(f'Number of Threads: {n}')
    start = time.time()
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=n, cv=kfold)   # grid search
    grid_result = grid_search.fit(X, label_encoded_y)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
      	print("%f (%f) with: %r" % (mean, stdev, param))

    scores = numpy.array(means).reshape(len(learning_rate), len(number_of_trees))
    elapsed = time.time() - start

    print(f'Elapsed Duration: {elapsed}')

scores = numpy.array(means).reshape(len(learning_rate), len(number_of_trees))
for i, value in enumerate(learning_rate):
    pyplot.plot(number_of_trees, scores[i], label = 'learning_rate: ' + str(value))
pyplot.legend()
pyplot.xlabel('number_of_trees')
pyplot.ylabel('Log Loss')
pyplot.savefig('number_of_trees_vs_learning_rate.png')

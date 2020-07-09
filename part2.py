from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import time
from matplotlib import pyplot

data = read_csv('train.csv')	# load data
dataset = data.values

X = dataset[:,0:94]	# split data into X and y
y = dataset[:,94]
label_encoded_y = LabelEncoder().fit_transform(y)	# encode string class values as integers
n_estimators = [100, 200, 300, 400, 500]
max_depth = [1]
subsample = [1.0]
colsample_bytree = [0.18]
learning_rate = [0.1, 0.01, 0.001, 0.0001]
gamma = [0] 
objective = ['binary:logistic']
reg_alpha = [0.2]
reg_lambda = [0.4]

param_grid = {'reg_lambda': reg_lambda,'reg_alpha': reg_alpha, 'objective': objective, 'gamma': gamma, 'colsample_bytree': colsample_bytree, 'subsample': subsample, 'max_depth': max_depth, 'n_estimators': n_estimators, 'learning_rate': learning_rate}

kfold_cross_validation = StratifiedKFold(n_splits = 10)

# results = []
num_threads = [2, 1, -1]
for n in num_threads:
    print(f'Number of Threads: {n} with {kfold_cross_validation} KFolds')
    start = time.time()
    best_grid_search = GridSearchCV(estimator = XGBClassifier(nthread = n), param_grid = param_grid, cv = kfold_cross_validation, scoring = 'neg_log_loss', n_jobs = n)
    grid_result = best_grid_search.fit(X, label_encoded_y)
    
    elapsed = time.time() - start
    # print(f'Param Grid: {param_grid}')
    # print(f'N-Estimator: {param_grid['n_estimators']}')
    # print(f'Learing Rate: {param_grid.get('learning_rate')}')
    print(f'Elapsed Duration: {elapsed}')
    # results.append((num_threads, elapsed))
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# print(results)
# plot results
# pyplot.plot(num_threads, results)
# pyplot.ylabel('Speed (seconds)')
# pyplot.xlabel('Number of Threads')
# pyplot.title('XGBoost Training Speed vs Number of Threads')
# pyplot.show()

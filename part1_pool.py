import matplotlib.pyplot as plt                                 # helps to plot data
from sklearn import svm                                         # native support vector machine library in python
import multiprocessing                                          # native library import for python multiprocessing
import numpy as np                                              # efficient big data structure in python
from time import time                                           # measure algorithm performance using wall clocks

def plot_digits(X, y):
    '''Plot some of the digits'''
    fig = plt.figure(figsize = (8, 6))
    fig.tight_layout()
    for i in range(0, 20):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.imshow(X[i].reshape((8,8)), cmap = "Greys", vmin = 0, vmax = 16)
    plt.show()

def cvkfold(X, y, tuning_params, partitions, k):
    '''uses kfolds to cross validate training and test datasets'''
    n_tuning_params = tuning_params.shape[0]                                # n_tuning_params = 10 (default)
    partition = partitions[k]                                               # for each k-fold round in the 'cvkfold' loop takes the random 'partitions' array passed in    
    
    #  X.shape[0] -> number of rows
    # np.arange(0, X.shape[0]) -> 0,1,.....3,823
    Train = np.delete(np.arange(0, X.shape[0]), partition)                  # unions the other partitions into the training set and removes the test partition set
    
    Test = partition                                                        # takes the k-th position (1 - 5) as the test partition set  
    X_train = X[Train, :]                                                   # puts 'Train' partition set rows and shallow-copy all its columns in 'X_train' variable
    y_train = y[Train]
    X_test = X[Test, :]
    y_test = y[Test]
    accuracies = np.zeros(n_tuning_params)                                  # initializes 'accuracies' array with zeroes -> array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
    for i in range(0, n_tuning_params):                                     # for i in range(0, 10):
        svc = svm.SVC(C = tuning_params[i], kernel = "linear")              # linear support vector classifier returns a "best fit" hyperplane that divides your data
        accuracies[i] = svc.fit(X_train, y_train).score(X_test, y_test)     # trains the liner hyperplane on the x and y training partition and tests it on the test partition for each loop iteraion 1-10
    return accuracies

def parallel_tuning_pool(cpu_count, kfolds, X, y):
    '''uses parallel cpus to cross validate best hyperplane with pool'''
    tuning_params = np.logspace(-6, -1, 10)                                 # tuning_params = array([1.00000000e-06,3.59381366e-06,1.29154967e-05,4.64158883e-05,1.66810054e-04,5.99484250e-04,2.15443469e-03,7.74263683e-03,2.78255940e-02,1.00000000e-01])
    partitions = np.array_split(np.random.permutation([i for i in range(0, X.shape[0])]), kfolds)
    pool = multiprocessing.Pool(cpu_count)
    expression_list = [(X, y, tuning_params, partitions, k) for k in range(0, kfolds)] # loops for each of 5 kfolds 
    Accuracies = np.array(pool.starmap(cvkfold, expression_list))
    pool.close()
    CV_accuracy = np.mean(Accuracies, axis = 0)                             # returns the 10 accuracy iteration runs on test data from the support vector classfier hyperplane garnered from training
    best_tuning_param = tuning_params[np.argmax(CV_accuracy)]               # takes the c parameters that gives the best out of the 10 returned accuracies
    return best_tuning_param

if __name__ == "__main__":
    test = np.loadtxt("optdigits.txt", delimiter = ",")
    X = test[:, 0:64]                                                       # X = 3,823 shallow-copy rows from "optdigits.txt" and 1-64 column features from "optdigits.txt"
    y = test[:, 64]                                                         # y = 3,823 shallow-copy rows from "optdigits.txt" and the last(64th) column from "optdigits.txt"
    cpu_count = 1
    kfolds = 5
    t1 = time()                                                             # starts the timer for 1 cpu test
    best_tuning_param = parallel_tuning_pool(cpu_count, kfolds, X, y)
    print('Best tuning param %0.6f.'% best_tuning_param)
    print('Serial runs in %0.3f seconds.' % (time() - t1))                  # prints the wall clock time of 1 cpu cross validation hyperparameter tuning
    cpu_count = 2
    t2 = time()                                                             # starts the timer for 2 cpu test
    best_tuning_param_2 = parallel_tuning_pool(cpu_count, kfolds, X, y)
    print('Best tuning param %0.6f.'% best_tuning_param_2)
    print('2 CPU Pool runs in %0.3f seconds.' % (time() - t2))              # prints the wall clock time of 2 cpu cross validation hyperparameter tuning
    cpu_count = 4
    t4 = time()                                                             # starts the timer for 4 cpu test
    best_tuning_param_4 = parallel_tuning_pool(cpu_count, kfolds, X, y)
    print('Best tuning param %0.6f.'% best_tuning_param_4)
    print('4 CPU Pool runs in %0.3f seconds.' % (time() - t4))              # prints the wall clock time of 2 cpu cross validation hyperparameter tuning
    cpu_count = 8
    t8 = time()                                                             # starts the timer for 8 cpu test
    best_tuning_param_8 = parallel_tuning_pool(cpu_count, kfolds, X, y)
    print('Best tuning param %0.6f.'% best_tuning_param_8)
    print('8 CPU Pool runs in %0.3f seconds.' % (time() - t8))              # prints the wall clock time of 2 cpu cross validation hyperparameter tuning
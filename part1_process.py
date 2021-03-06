import multiprocessing, os										
import matplotlib
matplotlib.use('Agg')                                           # prevents matplotlib error on discovery cluster
from matplotlib import pyplot                                   # helps to plot data
from sklearn import svm                                         
import numpy as np                                              
from time import time
import json                                           

def cvkfold(X, y, tuning_params, partitions, k):
    '''uses kfolds to cross validate training and test datasets'''
    n_tuning_params = tuning_params.shape[0]                                # n_tuning_params = 10 (default)
    partition = partitions[k]                                               # for each k-fold round in the 'cvkfold' loop takes the random 'partitions' array passed in    
    
    # X.shape[0] -> number of rows
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
    return(accuracies)


def parallel_tuning_process(kfolds, X, y):
    '''uses parallel cpus to cross validate best hyperplane with process'''
    tuning_params = np.logspace(-6, -1, 10)                                
    partitions = np.array_split(np.random.permutation([i for i in range(0, X.shape[0])]), kfolds)

    process_jobs = []

    for k in range(0, kfolds):   
        pool = multiprocessing.Process(target = cvkfold, args = (X, y, tuning_params, partitions, k))
        process_jobs.append(pool)

    for process_job in process_jobs:
        process_job.start()

    for process_job in process_jobs:
        process_job.join()

if __name__=='__main__':

    test = np.loadtxt("optdigits.txt", delimiter = ",")
    X = test[:, 0:64]                                                       # X = 3,823 shallow-copy rows from "optdigits.txt" and 1-64 column features from "optdigits.txt"
    y = test[:, 64]                                                         # y = 3,823 shallow-copy rows from "optdigits.txt" and the last(64th) column from "optdigits.txt"
    kfolds = 5
    t1 = time()
    parallel_tuning_process(kfolds, X, y)
    # best_tuning_param = parallel_tuning_process(spawn_count, kfolds, X, y)     
    # print('Best tuning param %0.6f.'% best_tuning_param)
    elapsed = (time() - t1)
    print('Process runs in %0.9f seconds.' % (elapsed))                  # prints the wall clock time of 1 cpu cross validation hyperparameter tuning
    results_dict = {'Cpu Count':  multiprocessing.cpu_count(), 'Runtime': elapsed}
    print(results_dict)
    cpu_count = [8, 7, 6, 5, 4, 3, 2, 1]
    times_elapsed = [5.291589260101318, 5.274212121963501, 5.3853960037231445, 5.350943326950073, 7.574889183044434, 8.657461166381836, 11.808668375015259, 26.79225444793701]
    pyplot.plot(cpu_count, times_elapsed, label = 'multiprocessing.Process wall clock times')
    pyplot.legend()
    pyplot.xlabel('CPU Count')
    pyplot.ylabel('Runtimes ')
    pyplot.savefig('cpu_count_x_axis_vs_mp_process_runtimes_y_axis')
    with open('runtime.txt', 'a+') as outfile:
        json.dump(results_dict, outfile)

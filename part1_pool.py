from sklearn import svm                                         # native support vector machine library in python
import multiprocessing                                          # native library import for python multiprocessing
import numpy as np                                              # efficient big data structure in python
from time import time                                           # measure algorithm performance using wall clocks
import matplotlib
matplotlib.use('Agg')                                           # prevents matplotlib error on discovery cluster
from matplotlib import pyplot                                   # helps to plot data

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
    t1_start = time()                                                             # starts the timer for 1 cpu test
    best_tuning_param = parallel_tuning_pool(cpu_count, kfolds, X, y)
    t1_duration = time() - t1_start
    print('Best tuning param %0.6f.'% best_tuning_param)
    print('Serial runs in %0.3f seconds.' % (t1_duration))                  # prints the wall clock time of 1 cpu cross validation hyperparameter tuning
    
    cpu_count = 2
    t2_start = time()                                                             
    best_tuning_param_2 = parallel_tuning_pool(cpu_count, kfolds, X, y)
    t2_duration = time() - t2_start
    print('Best tuning param %0.6f.'% best_tuning_param_2)
    print('2 CPU Pool runs in %0.3f seconds.' % (t2_duration))             
    
    cpu_count = 3
    t3_start = time()                                                             
    best_tuning_param_3 = parallel_tuning_pool(cpu_count, kfolds, X, y)
    t3_duration = time() - t3_start
    print('Best tuning param %0.6f.'% best_tuning_param_3)
    print('3 CPU Pool runs in %0.3f seconds.' % (t3_duration))              
     
    cpu_count = 4
    t4_start = time()                                                             
    best_tuning_param_4 = parallel_tuning_pool(cpu_count, kfolds, X, y)
    t4_duration = time() - t4_start
    print('Best tuning param %0.6f.'% best_tuning_param_4)
    print('4 CPU Pool runs in %0.3f seconds.' % (t4_duration))              
    
    cpu_count = 5
    t5_start = time()                                                             
    best_tuning_param_5 = parallel_tuning_pool(cpu_count, kfolds, X, y)
    t5_duration = time() - t5_start
    print('Best tuning param %0.6f.'% best_tuning_param_5)
    print('5 CPU Pool runs in %0.3f seconds.' % (t5_duration))        

    cpu_count = 6
    t6_start = time()                                                             
    best_tuning_param_6 = parallel_tuning_pool(cpu_count, kfolds, X, y)
    t6_duration = time() - t6_start
    print('Best tuning param %0.6f.'% best_tuning_param_6)
    print('6 CPU Pool runs in %0.3f seconds.' % (t6_duration))    

    cpu_count = 7
    t7_start = time()                                                             
    best_tuning_param_7 = parallel_tuning_pool(cpu_count, kfolds, X, y)
    t7_duration = time() - t7_start
    print('Best tuning param %0.6f.'% best_tuning_param_7)
    print('7 CPU Pool runs in %0.3f seconds.' % (t7_duration))         

    cpu_count = 8
    t8_start = time()                                                             # starts the timer for 8 cpu test
    best_tuning_param_8 = parallel_tuning_pool(cpu_count, kfolds, X, y)
    t8_duration = time() - t8_start
    print('Best tuning param %0.6f.'% best_tuning_param_8)
    print('8 CPU Pool runs in %0.3f seconds.' % (t8_duration))              # prints the wall clock time of 8 cpu cross validation hyperparameter tuning
        
    cpu_count = [1, 2, 3, 4, 5, 6, 7, 8]
    times_elapsed = [t1_duration, t2_duration, t3_duration, t4_duration, t5_duration, t6_duration, t7_duration, t8_duration]
    pyplot.plot(cpu_count, times_elapsed, label = 'multiprocessing.Pool wall clock times: ')
    pyplot.legend()
    pyplot.xlabel('CPU Count')
    pyplot.ylabel('Runtimes ')
    pyplot.savefig('cpu_count_x_axis_vs_mp_pool_runtimes_y_axis')
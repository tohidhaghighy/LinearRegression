import math

import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from lrm import LogisticRegressionClassifier

def cross_validate(x, y, k=5):
    # Stacking x and y horiontally
    m = np.hstack((x, y.reshape(x.shape[0],-1)))
    # Shuffling data to randomize their order
    np.random.shuffle(m)
    # Splitting x and y
    x = m[:, :-1]
    y = m[:, -1].reshape(x.shape[0],-1)
    dl = len(y)
    fl = int(dl/k)
    folds_indices = [(i*fl, (i+1)*fl) for i in range(0, k)]
    scores = []
    for i in range(0, k):
        i, j = folds_indices[i]
        test_x = x[i:j, :]
        test_y = y[i:j, :]
        train_x = np.vstack((x[0:i, :], x[j:, :]))
        train_y = np.vstack((y[0:i, :], y[j:, :]))
        lrc_model = LogisticRegressionClassifier()
        lrc_model.fit(train_x, train_y)
        s = lrc_model.score(test_x, test_y)
        scores.append(s)
    return sum(scores) / len(scores)


def create_mini_batches(X, y, batch_size): 
    mini_batches = [] 
    data = np.hstack((X, y)) 
    np.random.shuffle(data) 
    n_minibatches = data.shape[0] // batch_size 
    
    for i in range(n_minibatches + 1): 
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    if data.shape[0] % batch_size != 0: 
        mini_batch = data[i * batch_size:data.shape[0]] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    
    return mini_batches 
  

def gradientDescent(X, y, learning_rate = 0.0000001, batch_size = 32): 
    theta = np.zeros((X.shape[1], 1)) 
    error_list = [] 
    max_iters = 10000
    for itr in range(max_iters): 
        mini_batches = create_mini_batches(X, y, batch_size) 
        for mini_batch in mini_batches: 
            X_mini, y_mini = mini_batch 
            theta =  theta - learning_rate * LogisticRegressionClassifier.gradient_descent(X_mini, y_mini, theta)  
            error_list.append(LogisticRegressionClassifier.cost(X_mini, y_mini, theta)) 
  
    return theta, error_list 


def adagrad(X, y, learning_rate = 0.000001, batch_size = 32,fudge_factor = 1e-6): 
    theta = np.zeros((X.shape[1], 1))
    tt = np.zeros((X.shape[1], 1))
    error_list = [] 
    max_iters = 1
    for itr in range(max_iters): 
        mini_batches = create_mini_batches(X, y, batch_size) 
        for mini_batch in mini_batches: 
            X_mini, y_mini = mini_batch 
            tt+=tt**2
            theta =  theta - learning_rate * (LogisticRegressionClassifier.gradient_descent(X_mini, y_mini, theta))/(fudge_factor + np.sqrt(tt))
            error_list.append(LogisticRegressionClassifier.cost(X_mini, y_mini, theta)) 
  
    return theta, error_list 


def Show_Image(x,y,norm_x):
    lrc_normal = LogisticRegressionClassifier()
    lrc_normal.fit(x, y)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.set(xlabel="Iteration", ylabel="Cost", title="Unchanged Features")
    lrc_normal.plot_cost_list(ax)
    fig.savefig("lrc_normal.png")
    plt.close(fig)
    print("Unchanged Features - Misclassification Rate", cross_validate(x, y, k=5))
    print("Unchanged Features - iterations", len(lrc_normal.cost_list))

    lrc_std = LogisticRegressionClassifier()
    lrc_std.fit(norm_x, y)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.set(xlabel="Iteration", ylabel="Cost", title="Standardized Features")
    lrc_std.plot_cost_list(ax)
    fig.savefig("lrc_std.png")
    plt.close(fig)
    print("Standardized Features - Misclassification Rate:", cross_validate(x_std, y, k=5))
    print("Standardized Features - iterations:", len(lrc_std.cost_list))


def Show_normal_Image(x,y):
    lrc_std = LogisticRegressionClassifier()
    lrc_std.fit(x, y)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.set(xlabel="Iteration", ylabel="Cost", title="Standardized Features")
    lrc_std.plot_cost_list(ax)
    fig.savefig("lrc_std.png")
    plt.close(fig)
    print("Standardized Features - Misclassification Rate:", cross_validate(x_std, y, k=5))
    print("Standardized Features - iterations:", len(lrc_std.cost_list))


if(__name__=="__main__"):

    #main
    names = ['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
 'LOR', 'CGPA', 'Research', 'Chance of Admit']

    df = pd.read_csv("Admission_Predict.csv", header=None, names=names)
    

    #df_features = df.iloc[1:, 1:8]
    #df_targets = df.loc[: , "Chance of Admit"]

    x = df.iloc[1:, 1:8].values
    y = df.loc[1: , "Chance of Admit"].values
    print(x.astype(float))
    print(y.astype(float))
    
    x=x.astype(float)
    y=y.astype(float)


    X_train = x[:350, :-1]
    print(X_train)
    y_train = y[:350].reshape((-1, 1))
    print(y_train)
    X_test = x[350:, :-1] 
    y_test = y[350:].reshape((-1, 1)) 

    # Standardizing the features onto unit scale (mean = 0 and variance = 1)
    x_std = StandardScaler().fit_transform(X_train)

    
    Show_Image(X_train,y_train,x_std)
    
    gradientDescent(X_train,y_train)
    


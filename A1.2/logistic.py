import numpy as np
import pandas as pd
import sys
from scipy.special import softmax
from sklearn import preprocessing


'''def loss(X : np.array, y : np.array, w : np.array, counts : np.array):
    
    lt1 = 0
    lt2=0
    m = X@w
    n = y.shape[0]
    c = w.shape[1]
    C = np.array([0, 1, 2, 3])
    
    for i in range(n):
        pos = int(y[i]@C.T)
        lt1+= m[i]@(y[i].T)
        # sum=0
        # for k in range(c):
        #     sum+= np.exp(m[i][k])
        lt2 += np.log(np.sum(np.exp(m[i])))/counts[pos]

    l = (1/(2*n))*(-lt1 + lt2)
    return l''' 

def loss(X : np.array, y : np.array, w : np.array, counts : np.array):
    n = y.shape[0]
    P = softmax(X@w, axis=1)
    P = np.log(P)
    P = P*y
    P = P@((1/counts).T)
    return (-1/(2*n))*np.sum(P)

def ternary_search(eta_0 : np.float64, gradient : np.array, X_train: np.array, y_train : np.array, w : np.array, counts : np.array):
    eta_l = 0
    eta_h = eta_0
    l = loss(X_train, y_train, w, counts)
    while(l > loss(X_train, y_train, w-eta_h*gradient, counts)):
        eta_h = 2*eta_h
        

    for i in range(20):
        
        eta_1 = (2*eta_l + eta_h)/3
        eta_2 = (2*eta_h + eta_l)/3
        l1 = loss(X_train, y_train, w-eta_1*gradient, counts)   
        l2 = loss(X_train, y_train, w-eta_2*gradient, counts)
        if(l1 > l2):
            eta_l = eta_1
        elif(l1 < l2):
            eta_h = eta_2
        else:
            eta_l = eta_1
            eta_h = eta_2

    # print("hello1")
    eta = (eta_l + eta_h)/2
    return eta

def X_change(X_org : np.array, y_org : np.array, counts : np.array, uni_feat : int):
    n = y_org.shape[0]
    X_new = np.zeros(X_org.shape, dtype=np.float64)
    for i in range(n):
        X_new[i] = X_org[i]/counts[int((y_org[i] -1)%uni_feat)]

    #print(X_new)
    
    return X_new

def one_hot_encoding(y_org : np.array, uni_feat : int):
    n = y_org.shape[0]
    y_new = np.zeros((n, uni_feat), dtype= np.float64)
    
    for i in range(n):
        pos = int((y_org[i]-1))%uni_feat
        y_new[i][pos] = 1
    
    return y_new

'''def softmax_matrix(X : np.array, y : np.array, w : np.array, counts : np.array, uni_feat : int):
    n = y.shape[0]
    #m = y.shape
    #print(m)
    P = np.zeros((n, uni_feat), dtype= np.float64)
    c = np.array([0, counts[1], 2*counts[2], 3*counts[3]])
    for i in range(n):
        
        pos = int(y[i]@c.T)
        #pos = int(y[i]-1)
        freq = counts[pos]
        p_nums = np.exp(X[i]@w)
        p_den = np.sum(np.exp(X[i]@w))
        for j in range(uni_feat):
            p = p_nums[j]/p_den
            p = p/freq
            P[i][j] = p
         
    return P'''

def mini_batch_gradient_const(n : int, epochs : int, batch_size : int, learning_rate : np.float64, X : np.array, y : np.array, weights : np.array, X_org : np.array):
    no_of_batches = int((n+batch_size-1)/batch_size)
    for i in range(epochs):
        for j in range(no_of_batches):
            X_train = X[j*batch_size : min((j+1)*batch_size, n)]
            y_train = y[j*batch_size : min((j+1)*batch_size, n)]
            X_train_org = X_org[j*batch_size : min((j+1)*batch_size, n)]
            #P = softmax_matrix(X_train, y_train, weights, counts, unique_values_size)
            P = softmax(X_train_org@weights, axis = 1)
            bs = X_train.shape[0]
            gradient = (1/(2*bs))*(X_train.T@(-y_train + P))
            weights = weights - learning_rate*gradient
            
    return weights

def mini_batch_gradient_adaptive(n : int, epochs : int, batch_size : int, eta_0 : np.float64, k_0 : np.float64, X : np.array, y : np.array, weights : np.array, X_org : np.array):
    no_of_batches = int((n+batch_size-1)/batch_size)
    for i in range(epochs):
        eta = eta_0/(1 + k_0*(i+1))
        for j in range(no_of_batches):
            X_train = X[j*batch_size : min((j+1)*batch_size, n)]
            y_train = y[j*batch_size : min((j+1)*batch_size, n)]
            X_train_org = X_org[j*batch_size : min((j+1)*batch_size, n)]
            #P = softmax_matrix(X_train, y_train, weights, counts, unique_values_size)
            P = softmax(X_train_org@weights, axis = 1)
            bs = X_train.shape[0]
            gradient = (1/(2*bs))*(X_train.T@(-y_train + P))
            weights = weights - eta*gradient
    
    return weights

def mini_batch_gradient_line(n : int, epochs : int, batch_size : int, eta_0 : np.float64, X : np.array, y : np.array, weights : np.array, counts : np.array, X_org : np.array):
    no_of_batches = int((n+batch_size-1)/batch_size) 
    for i in range(epochs):
        for j in range(no_of_batches):
            X_train = X[j*batch_size : min((j+1)*batch_size, n)]
            y_train = y[j*batch_size : min((j+1)*batch_size, n)]
            X_train_org = X_org[j*batch_size : min((j+1)*batch_size, n)]
            #P = softmax_matrix(X_train, y_train, weights, counts, unique_values_size)
            P = softmax(X_train_org@weights, axis = 1)
            bs = X_train.shape[0]
            gradient = (1/(2*bs))*(X_train.T@(-y_train + P))
            eta = ternary_search(eta_0, gradient, X_train_org,  y_train, weights, counts)
            weights = weights - eta*gradient
        
    return weights

def mini_batch_gradient_adam(n : int, epochs : int, batch_size : int, X : np.array, y : np.array, weights : np.array, X_org : np.array, counts : np.array):
    alpha = 0.001
    beta_1 = 0.91
    beta_2 = 0.999
    m = 0
    v = 0
    epsilon = 1e-8
    no_of_batches = int((n+batch_size-1)/batch_size)
    for i in range(epochs):
        alpha_t = alpha*(1-np.power(beta_2, i+1))/np.power(1-np.power(beta_1, i+1), 2)
        for j in range(no_of_batches):
            X_train = X[j*batch_size : min((j+1)*batch_size, n)]
            y_train = y[j*batch_size : min((j+1)*batch_size, n)]
            X_train_org = X_org[j*batch_size : min((j+1)*batch_size, n)]
            #P = softmax_matrix(X_train, y_train, weights, counts, unique_values_size)
            P = softmax(X_train_org@weights, axis = 1)
            bs = X_train.shape[0]
            gradient = (1/(2*bs))*(X_train.T@(-y_train + P))
            
            m = beta_1*m + (1-beta_1)*gradient
            v = beta_2*v + (1-beta_2)*(gradient*gradient)
            
            modified_gradient = m/(np.sqrt(v) + epsilon*(np.power(1-np.power(beta_2, i+1), 0.5)))
            weights = weights - alpha_t*modified_gradient
        
        if(i%5==0):
            print(loss(X_org, y_train_ohe, weights, counts))
            print(weights)
        

            
    return weights

part = sys.argv[1]
if(part == 'a'):
    train_file = sys.argv[2]
    parameter = sys.argv[3]
    modelweight = sys.argv[4]
    
    df_train_origial = pd.read_csv(train_file)
    X_train_original = df_train_origial.drop(columns= 'Race').to_numpy(dtype=np.float64)
    y_train_original = df_train_origial['Race'].to_numpy(dtype=np.float64)
    X_train_withbias = np.insert(X_train_original, 0, 1, axis=1)
    unique_values, counts = np.unique(y_train_original, return_counts=True)
    counts = np.array(counts, dtype = np.float64)
    unique_values_size = unique_values.size
    X_train_final = X_change(X_train_withbias, y_train_original, counts, unique_values_size)
    no_of_features = X_train_withbias.shape[1]
    y_train_ohe = one_hot_encoding(y_train_original, unique_values_size)
    weights = np.zeros((no_of_features, unique_values_size), dtype= np.float64)
    n = y_train_ohe.shape[0]



    f = open(parameter, "r")
    lines = f.readlines()
    f.close()
    strategy = int(lines[0])
    epochs = int(lines[2])
    batch_size = int(lines[3])
    if(strategy == 1):
        learning_rate = np.array([lines[1]], dtype= np.float64)
        learning_rate = learning_rate[0]
        weights = mini_batch_gradient_const(n, epochs, batch_size, learning_rate, X_train_final, y_train_ohe, weights,X_train_withbias)
    elif(strategy == 2):
        eta_0, k_0 = lines[1].split(',')
        eta_0 = float(eta_0)
        k_0 = float(k_0)
        weights = mini_batch_gradient_adaptive(n, epochs, batch_size, eta_0, k_0, X_train_final, y_train_ohe, weights, X_train_withbias)
    else:
        eta_0 = float(lines[1])
        weights = mini_batch_gradient_line(n, epochs, batch_size, eta_0, X_train_final, y_train_ohe, weights, counts, X_train_withbias)

    flattened_weights = weights.flatten()
    np.savetxt(modelweight, flattened_weights)

else:
    train_file = sys.argv[2]
    test_file = sys.argv[3]
    modelweight = sys.argv[4]
    modelprediction = sys.argv[5]

    df_train_origial = pd.read_csv(train_file)
    X_train_original = df_train_origial.drop(columns= 'Race').to_numpy(dtype=np.float64)
    y_train_original = df_train_origial['Race'].to_numpy(dtype=np.float64)
    X_train_withbias = np.insert(X_train_original, 0, 1, axis=1)
    scaler = preprocessing.StandardScaler().fit(X_train_withbias)
    X_train_withbias = scaler.transform(X_train_withbias)
    unique_values, counts = np.unique(y_train_original, return_counts=True)
    counts = np.array(counts, dtype = np.float64)
    unique_values_size = unique_values.size
    X_train_final = X_change(X_train_withbias, y_train_original, counts, unique_values_size)
    no_of_features = X_train_withbias.shape[1]
    y_train_ohe = one_hot_encoding(y_train_original, unique_values_size)
    weights = np.zeros((no_of_features, unique_values_size), dtype= np.float64)
    n = y_train_ohe.shape[0]
    weights = mini_batch_gradient_adam(n, 30, 500, X_train_final, y_train_ohe, weights, X_train_withbias, counts)
    flattened_weights = weights.flatten()
    np.savetxt(modelweight, flattened_weights)

    df_test_origial = pd.read_csv(test_file)
    X_test_original = df_test_origial.to_numpy()
    X_test_withbias = np.insert(X_test_original, 0, 1, axis=1)
    X_test_withbias = scaler.transform(X_test_withbias)

    y_pred = X_test_withbias@weights
    y_pred = softmax(y_pred, axis=1)
    np.savetxt(modelprediction, y_pred, delimiter=",")


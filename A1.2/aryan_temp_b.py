import numpy as np
import pandas as pd

from scipy.special import softmax
from sklearn import preprocessing
import sys

def loss(X : np.array, y : np.array, w : np.array, counts : np.array):
    n = y.shape[0]
    P = softmax(X@w, axis=1)
    P = np.log(P)
    P = P*y
    P = P@((1/counts).T)
    return (-1/(2*n))*np.sum(P)

def one_hot_encoding(y_org : np.array, uni_feat : int):
    n = y_org.shape[0]
    y_new = np.zeros((n, uni_feat), dtype= np.float64)
    
    for i in range(n):
        pos = int((y_org[i]-1))%uni_feat
        y_new[i][pos] = 1
    
    return y_new

def X_change(X_org : np.array, y_org : np.array, counts : np.array, uni_feat : int):
    n = y_org.shape[0]
    X_new = np.zeros(X_org.shape, dtype=np.float64)
    for i in range(n):
        X_new[i] = X_org[i]/counts[int((y_org[i] -1)%uni_feat)]

    #print(X_new)
    
    return X_new

def mini_batch_gradient_adam(n : int, epochs : int, batch_size : int, X : np.array, y : np.array, weights : np.array, X_org : np.array, counts : np.array):
    alpha = 0.001
    beta_1 = 0.91
    beta_2 = 0.999
    m = 0
    v = 0
    epsilon = 1e-7
    no_of_batches = int((n+batch_size-1)/batch_size)
    for i in range(epochs):
        alpha = alpha*(1-np.power(beta_2, i+1))/np.power(1-np.power(beta_1, i+1), 2)
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
            weights = weights - alpha*modified_gradient
        
        print(loss(X_org, y_train_ohe, weights, counts))
        print(weights)
        

            
    return weights

train_file = sys.argv[1]
test_file = sys.argv[2]
#parameter = sys.argv[2]
#modelweight = sys.argv[3]
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


df_test_origial = pd.read_csv(test_file)
X_test_original = df_test_origial.to_numpy()
X_test_withbias = np.insert(X_test_original, 0, 1, axis=1)
X_test_withbias = scaler.transform(X_test_withbias)

y_pred = X_test_withbias@weights
y_pred = softmax(y_pred, axis=1)
np.savetxt("output.csv", y_pred, delimiter=",")

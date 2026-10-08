import numpy as np
import pandas as pd
import json
import sys
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

with open('mapping.json', 'r') as json_file:
    data_dict = json.load(json_file)

def ohe(df, target : list):
    
    for key in data_dict.keys():
        print(key)
        if (key in target):
            continue
        possible_values = data_dict[key]
        one_hot = pd.get_dummies(df[key])
        one_hot = one_hot.reindex(columns=sorted(possible_values),fill_value=False)
        one_hot = one_hot.iloc[:,1:]
        one_hot = one_hot.astype(int)
        one_hot.columns = [f"{key}_{col}" for col in one_hot.columns]
        one_hot_array = one_hot.values
        column_index = df.columns.get_loc(key)
        df = df.drop(columns=[key])
        df_array = df.values
        columns_before = df_array[:, :column_index]
        columns_after = df_array[:, column_index:]
        new_columns = list(df.columns[:column_index]) + list(one_hot.columns) + list(df.columns[column_index:])
        combined_array = np.hstack([df_array[:, :column_index], one_hot_array, df_array[:, column_index:]])
        df = pd.DataFrame(combined_array, columns=new_columns)
        print(df.shape)
    
    return df

def sigmoid(y : np.array):
    return (1/(1+np.exp(-y)))

def loss(X : np.array, w : np.array, y : np.array):
    n = y.shape[0]
    y_hat = X@w.T
    return (1/n)*np.sum(np.log(1 + np.exp(-y*y_hat)))

def gradient(X : np.array, w : np.array, y : np.array):
    y_hat = X@w.T
    return X.T@(-y*(1-sigmoid(y*y_hat)))

def mini_batch(epochs : int, batch_size : int, X : np.array, y : np.array, weights : np.array):
    alpha = 0.01
    beta_1 = 0.9
    beta_2 = 0.99
    m = 0
    v = 0
    epsilon = 1e-8
    n = y.shape[0]
    no_of_batches = int((n+batch_size-1)/batch_size)
    for i in range(epochs):
        alpha_t = alpha*(1-np.power(beta_2, i+1))/np.power(1-np.power(beta_1, i+1), 2)
        for j in range(no_of_batches):
            X_train = X[j*batch_size : min((j+1)*batch_size, n)]
            y_train = y[j*batch_size : min((j+1)*batch_size, n)]
            grad = gradient(X_train, weights, y_train)
            m = beta_1*m + (1-beta_1)*grad
            v = beta_2*v + (1-beta_2)*(grad*grad)
            
            modified_gradient = m/(np.sqrt(v) + epsilon*(np.power(1-np.power(beta_2, i+1), 0.5)))
            weights = weights - alpha_t*modified_gradient
        
        print(loss(X, weights, y))
    
    return weights
            

train_file = sys.argv[1]
#parameter = sys.argv[2]
#modelweight = sys.argv[3]
test_file = sys.argv[2]
df_train_origial = pd.read_csv(train_file)
df_train_origial = ohe(df_train_origial, ["Gender", ])
X_train_original = df_train_origial.drop(columns= 'Gender').to_numpy(dtype=np.float64)
y_train_original = df_train_origial['Gender'].to_numpy(dtype=np.float64)
X_train_withbias = np.insert(X_train_original, 0, 1, axis=1)
scaler  = StandardScaler().fit(X_train_withbias)
X_train_withbias = scaler.transform(X_train_withbias)
unique_values, counts = np.unique(y_train_original, return_counts=True)
counts = np.array(counts, dtype = np.float64)
unique_values_size = unique_values.size
#X_train_final = X_change(X_train_withbias, y_train_original, counts, unique_values_size)
no_of_features = X_train_withbias.shape[1]
#y_train_ohe = one_hot_encoding(y_train_original, unique_values_size)
weights = np.zeros(no_of_features, dtype= np.float64)
n = y_train_original.shape[0]

weights = mini_batch(25,800, X_train_withbias,y_train_original, weights)

df_test_origial = pd.read_csv(test_file)
df_test_origial = ohe(df_test_origial, "Gender")
X_test_original = df_test_origial.to_numpy()
X_test_withbias = np.insert(X_test_original, 0, 1, axis=1)
X_test_withbias = scaler.transform(X_test_withbias)

y_pred = sigmoid(X_test_withbias@weights)

f = open("out.txt", "w")
for i in y_pred:
    if(i >= 0.5):
        f.write('1' + '\n')
    else:
        f.write('-1\n')

f.close()    


import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.preprocessing import StandardScaler
import json
import sys

train_file = sys.argv[1]

# anystring is the name of the file without the extension and train_ prefix
anystring = train_file.split('.')[0].split('_')[1]

data = pd.read_csv(train_file)
# column headers would not be given in test data
X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1].to_numpy()

y = np.where(y == 0, -1, 1)

# Preprocess
scaler = StandardScaler()
X = scaler.fit_transform(X)

C = 1

# Define optimization variables
w = cp.Variable(X.shape[1])
b = cp.Variable()
epsi = cp.Variable(X.shape[0])

# Objective function
objective = 0.5 * cp.norm(w, 1) + C * cp.sum(epsi)

# Constraints
constraints = [y[i] * (X[i] @ w + b) >= 1 - epsi[i] for i in range(X.shape[0])]
constraints += [xi >= 0 for xi in epsi]

# Define and solve the optimization problem
prob = cp.Problem(cp.Minimize(objective), constraints)
prob.solve()

# Extract the optimal weights, bias, and slack variables
w_opt = w.value
b_opt = b.value
epsi_opt = epsi.value

dict1 = {'weights': w_opt.tolist(), 'bias': float(b_opt)}

dict2 = {'seperable': 0 , 'support_vectors': []}
y_pred = (X @ w_opt + b_opt)

# find if the data is linearly separable
if np.all(y == np.sign(y_pred)):
    print('Data is linearly separable')
    dict2['seperable'] = 1
    # find support vectors
    '''
        Support vectors are the data points that lie on the margin.
        func_margin = y * y_pred
        margin_dist = np.abs(func_margin - (1 - epsi_opt))
    '''
    margin_dist = np.abs(y_pred - (y - epsi_opt))
    sv_indices = np.where(margin_dist <= 1e-4)[0]
    dict2['support_vectors'] = sv_indices.tolist() 


#save in a json file
with open(f'weights_{anystring}.json', 'w') as f:
    json.dump(dict1, f, indent = 2)

with open(f'sv_{anystring}.json', 'w') as f:
    json.dump(dict2, f, indent = 2)
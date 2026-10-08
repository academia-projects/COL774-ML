import numpy as np
import pandas as pd
import sys

part = sys.argv[1]  # 'a' or 'b'
train_file = sys.argv[2]  # 'data/train.csv'
test_file = sys.argv[3]  # 'data/test.csv'
model_predictions_file = sys.argv[5]  # 'data/modelpredictions.txt'
model_weights_file = sys.argv[6]  # 'data/modelweights.txt'

if(part == "a"):
    sample_weights_file = sys.argv[4]
    #sample_weights_file = "data/sample_weights1.txt"
    df = pd.read_csv(train_file)
    X = df.drop(columns='Total Costs').to_numpy()
    y = df['Total Costs'].to_numpy()
    X_new = np.insert(X, 0, 1, axis=1)

    C = []
    f = open(sample_weights_file, "r")
    for i in f:
        C.append(float(i))
    f.close()

    X_copy = X_new.copy()
    count = 0
    for i in X_copy:
        X_copy[count] = np.multiply(X_copy[count] , C[count])
        count+=1

    count=0
    for i in y:
        y[count] = y[count]*C[count]
        count+=1
    
    w = np.dot(np.linalg.inv(np.dot(X_new.T, X_copy)),np.dot(X_new.T,y))
    
    f = open(model_weights_file, "w")
    for i in w:
        f.write(str(i) + '\n')
    f.close()

    df_test = pd.read_csv(test_file)
    X_test = df_test.to_numpy()
    X_new_test = np.insert(X_test, 0, 1, axis=1)
    y_pred = np.dot(X_new_test, w)

    f = open(model_predictions_file, "w")
    for i in y_pred:
        f.write(str(i)+ '\n')
    f.close()
    


else:
     
    regularization_file = sys.argv[4]  # 'data/regularization.txt'
    best_lambda_file = sys.argv[7]  # 'data/bestlambda.txt'
    df = pd.read_csv(train_file)
    X = df.drop(columns='Total Costs').to_numpy(dtype=np.float64)
    y_new = df['Total Costs'].to_numpy(dtype=np.float64)
    X_new = np.array(np.insert(X, 0, 1, axis=1), dtype=np.float64)
    X_new = X_new[:-4]
    y_new = y_new[:-4]
    n = X_new.shape[1]

    f = open(regularization_file, "r")
    lam = []
    for i in f:
        lam.append(np.float64(i))
    
    f.close()

    #print(X_new.shape)
    total_mse_list = [0]*len(lam)
    fold_size = len(y_new) // 10
    count=0
    for z in range(10):
        X_test = X_new[z*fold_size : (z+1)*fold_size]
        y_test = y_new[z*fold_size : (z+1)*fold_size]
        X_train = np.concatenate((X_new[:z*fold_size], X_new[(z+1)*fold_size: ]), axis=0, dtype=np.float64)
        y_train = np.concatenate((y_new[:z*fold_size], y_new[(z+1)*fold_size: ]), axis=0, dtype=np.float64)
        X_sq = np.matmul(X_train.T,X_train)
        Xy = np.matmul(X_train.T,y_train)
        
        for i in range(len(lam)):
            
            A = np.linalg.inv(X_sq + lam[i]*np.identity(n, dtype=np.float64))
            w = np.matmul(A,Xy)
            y_pred = np.matmul(X_test, w)
            diff = y_pred - y_test
              
            '''mse = 0
            for j in diff:
                mse = mse + j*j'''

            mse = np.square(diff).mean()
            total_mse_list[i] += mse
    
    
    
    minimum = total_mse_list[0]
    optimal_lambda = lam[0]
    for k in range(len(lam)):
        if(total_mse_list[k] < minimum):
            minimum = total_mse_list[k]
            optimal_lambda = lam[k]
    
    A = np.linalg.inv(np.matmul(X_new.T,X_new) + optimal_lambda*np.identity(n, dtype=np.float64))
    w_final = np.matmul(A,np.matmul(X_new.T,y_new))

    X_test_final = pd.read_csv(test_file).to_numpy(dtype=np.float64)
    X_test_final_new = np.array(np.insert(X_test_final, 0, 1, axis=1), dtype=np.float64)
    y_pred_final = np.matmul(X_test_final_new, w_final)
    np.savetxt(best_lambda_file, np.array([optimal_lambda]))# print(optimal_lambda)
    # f = open("data/predb.txt", "r")
    # pred = []
    # for i in f:
    #     pred.append(float(i))
    # f.close()

    f = open(model_weights_file, "w")
    for i in w_final:
        f.write(str(i)+ '\n')
    f.close()

    f = open(model_predictions_file, "w")
    for i in y_pred_final:
        f.write(str(i)+ '\n')
    f.close()


    # Pred = np.array(pred, dtype=np.float64)

    # error = Pred - y_pred_final
    # total_error=0

    # for i in range (len(error)):
    #     total_error += abs(error[i]/Pred[i])

    # total_error = total_error / len(error)
    # total_error = total_error*100

    # print(total_error)
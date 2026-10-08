import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge as rdg

cols=['Hospital County','Facility Name', 'Age Group', 'CCSR Diagnosis Code', 'CCSR Procedure Code', 'APR DRG Code', 'APR MDC Code', 'APR Severity of Illness Code', 'Birth Weight']

pd.set_option('future.no_silent_downcasting', True)     # dunno why necessary but is required

def preprocess(data):
    n = data.shape[0]
    m = data.shape[1]
    l = pd.Series(data['Zip Code - 3 digits']).value_counts()
    freq_encoded_zip=[]
    for i in range(n):
        freq_encoded_zip.append(l[data['Zip Code - 3 digits'][i]])
        

    data_trans = pd.get_dummies(data, columns=['Hospital Service Area', 'Gender', 'Race', 'Ethnicity', 'Type of Admission', 'Patient Disposition', 'APR Medical Surgical Description', 'Payment Typology 1', 'Payment Typology 2', 'Payment Typology 3', 'APR Risk of Mortality'])
    data_trans['freq_encoded_zip'] = freq_encoded_zip
    data_trans = data_trans.drop(columns=['Operating Certificate Number', 'Permanent Facility Id', 'CCSR Diagnosis Description', 'CCSR Procedure Description', 'APR DRG Description', 'APR MDC Description','APR Severity of Illness Description' ])
    return data_trans

def targ(data, test) :
    for col in cols:
        target_means = data.groupby(col)['Total Costs'].mean()
        data[col] = data[col].map(target_means)
        test[col] = test[col].map(lambda x: target_means.get(x, 0))
        # 0 ki jagah mean se replace kar de

df = pd.read_csv('data/train.csv') 
y = df['Total Costs'].copy()
print(df.shape)
# df.drop(columns = ['Total Costs'])
print("train shape initial : ")
print(df.shape)

df2 = pd.read_csv('data/test.csv')
print("test shape initial : ")
print(df2.shape)

df_trans = preprocess(df)
print("train shape fina; : ")
print(df_trans.shape)

test_trans = preprocess(df2)
targ(df_trans, test_trans)
print("test shape final : ")
print(test_trans.shape)


df_trans.to_csv('data/modified_train.csv', index = False)
test_trans.to_csv('data/modified_test.csv', index = False)
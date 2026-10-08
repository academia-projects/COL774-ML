import numpy as np
import pandas as pd
import json
import sys
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from scipy import stats


#encoder = OneHotEncoder(drop= 'first')
#Mean_per_group_dict = {}
pca = PCA(n_components=999)
#ratio = ['Hospital Service Area', 'Hospital County', 'Zip Code - 3 digits', 'Age Group', 'Patient Disposition']
#target_cols = ['Zip Code - 3 digits', 'Age Group', 'CCSR Diagnosis Code']
Counts = {}
with open('data/mapping.json', 'r') as json_file:
    data_dict = json.load(json_file)

def preprocess(data):
    n = data.shape[0]
    m = data.shape[1]
    #l = pd.Series(data['Zip Code - 3 digits']).value_counts()
    #freq_encoded_zip=[]
    ord_encoded_rom=[]
    ord_encoded_severity = []
    for i in range(n):
        
        #freq_encoded_zip.append(l[data['Zip Code - 3 digits'][i]])
        if(data['APR Risk of Mortality'][i] == 1):
            ord_encoded_rom.append(4)
        elif(data['APR Risk of Mortality'][i] == 2):
            ord_encoded_rom.append(3)
        elif(data['APR Risk of Mortality'][i] == 3):
            ord_encoded_rom.append(1)
        else:
            ord_encoded_rom.append(2)
        if(data['APR Severity of Illness Description'][i] == 1):
            ord_encoded_severity.append(4)
        elif(data['APR Severity of Illness Description'][i] == 2):
            ord_encoded_severity.append(3)
        elif(data['APR Severity of Illness Description'][i] == 3):
            ord_encoded_severity.append(1)
        else:
            ord_encoded_severity.append(2)

    data_new = data.copy(deep = True)
    data_new['ord_encoded_rom'] = ord_encoded_rom
    data_new['ord_encoded_severity'] = ord_encoded_severity
    #data_new = data_new.drop(columns=['CCSR Diagnosis Code', 'CCSR Procedure Code', 'APR DRG Code', 'APR MDC Code','APR Severity of Illness Code', 'Operating Certificate Number', 'Facility Name'])
    return data_new



def ohe(df, target : list):
    
    for key in data_dict.keys():
        #print(key)
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
        #df = df.drop(columns=[key])
        df_array = df.values
        columns_before = df_array[:, :column_index]
        columns_after = df_array[:, column_index:]
        new_columns = list(df.columns[:column_index]) + list(one_hot.columns) + list(df.columns[column_index:])
        combined_array = np.hstack([df_array[:, :column_index], one_hot_array, df_array[:, column_index:]])
        df = pd.DataFrame(combined_array, columns=new_columns)
        #print(df.shape)
    
    return df





train_file = sys.argv[1]
creation =  sys.argv[2]
selection = sys.argv[3]

f_create = open(creation, "w")
f_select = open(selection, "w")

df_train_original = pd.read_csv(train_file)

#df_train_original = Targ_Enc_train(df_train_original, target_cols, 'Total Costs')

#z_scores = np.abs(stats.zscore(df_train_original))
#df_train_original = df_train_original[(z_scores < 2.5).all(axis=1)]
df_train_original = ohe(df_train_original, ['Gender', 'APR Severity of Illness Description', 'APR Risk of Mortality', 'Emergency Department Indicator'])
#df_train_original = ratios_train(df_train_original)
#df_train_original = ohe(df_train_original, ['Gender'])
df_train_original = preprocess(df_train_original)
columns_org = df_train_original.drop(columns= 'Gender').columns.to_list()
columns_org.insert(0, 'bias')
print(len(columns_org))

X_train_original = df_train_original.drop(columns= 'Gender').to_numpy(dtype=np.float64)
y_train_original = df_train_original['Gender'].to_numpy(dtype=np.float64)
X_train_withbias = np.insert(X_train_original, 0, 1, axis=1)
scaler  = StandardScaler().fit(X_train_withbias)
X_train_withbias = scaler.transform(X_train_withbias)


X_train_withbias = pca.fit_transform(X_train_withbias)
comp = pca.components_
print(X_train_original.shape)

for i in columns_org:
    f_create.write(i + '\n')
    f_select.write('0\n')


for i in comp:
    c=0
    s = ""
    for j in i:
        s = s + str(j) + columns_org[c] + " + "
        c+=1
    
    f_create.write(s[:-3] + '\n')
    f_select.write('1\n')






f_create.close()
f_select.close()
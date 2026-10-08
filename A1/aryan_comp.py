import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoLars


train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

target_cols = ['Zip Code - 3 digits', 'Age Group', 'CCSR Diagnosis Description', 'APR DRG Description', 'APR MDC Description','CCSR Procedure Description', 'Permanent Facility Id']
encoder_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
Mean_per_group_dict = {}

def Targ_Enc_train(data, targ_cols, target):
    for col in targ_cols:
        mean_per_grp = data.groupby(col)[target].mean()
        Mean_per_group_dict[col] = mean_per_grp
        grps = data[col].unique()
        for grp in grps:
            data[col] = data[col].replace(grp, mean_per_grp[grp])

    return data

def Targ_enc_test(data, targ_cols):
    for col in targ_cols:
        mean_per_grp = Mean_per_group_dict[col]
        grps = data[col].unique()
        
        for grp in grps:
            if grp in mean_per_grp:
                data[col] = data[col].replace(grp, mean_per_grp[grp])
            else:
                data[col] = data[col].replace(grp, mean_per_grp.mean())
                

    return data


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
    data_new = data_new.drop(columns=['CCSR Diagnosis Code', 'CCSR Procedure Code', 'APR DRG Code', 'APR MDC Code', 'Operating Certificate Number', 'Facility Name'])
    return data_new

def erro(y_pred_final, Pred):
    error = Pred - y_pred_final
    error = abs(error)
    error.sort()
    error = error[:int(len(error)*(0.9))]
    total_error=error.T@error

    return total_error/len(error)


df = pd.read_csv(train_file)
df = Targ_Enc_train(df, target_cols, 'Total Costs')
df_feat = df.drop(columns='Total Costs')
y = df['Total Costs'].to_numpy()
df_trans = preprocess(df_feat)

features_to_encode = ['Hospital County', 'Hospital Service Area', 'Gender', 'Race', 'Ethnicity', 'APR Medical Surgical Description', 'Payment Typology 1', 'Payment Typology 2', 'Payment Typology 3', 'Patient Disposition', 'Type of Admission']
non_categorical_cols = [col for col in df_trans.columns if col not in features_to_encode]

train_encoded = encoder_ohe.fit_transform(df_trans[features_to_encode])
df_trans_sub = pd.DataFrame(train_encoded, columns=encoder_ohe.get_feature_names_out(features_to_encode))
df_trans = pd.concat([df_trans_sub, df_trans[non_categorical_cols].reset_index(drop=True)], axis=1)

X_train_t = df_trans.to_numpy()
#X_train_org = df_feat.to_numpy()

reg = Ridge(alpha= 40).fit(X_train_t, y)
#reg2 = Ridge(alpha= 40).fit(X_train_org, y)

df_test = pd.read_csv(test_file)
df_test = Targ_enc_test(df_test, target_cols)
df_test_trans = preprocess(df_test)
train_encoded = encoder_ohe.transform(df_test_trans[features_to_encode])
df_test_trans_sub = pd.DataFrame(train_encoded, columns=encoder_ohe.get_feature_names_out(features_to_encode))
df_test_trans = pd.concat([df_test_trans_sub, df_test_trans[non_categorical_cols].reset_index(drop=True)], axis=1)

X_test_trans = df_test_trans.to_numpy()
rows = X_test_trans.shape[0]
#X_test_org = df_test.to_numpy()


y_pred_t = reg.predict(X_test_trans)
#y_pred_or = reg2.predict(X_test_org)
y_pred_t = y_pred_t.reshape(rows)

f = open(output_file, "w")
for i in y_pred_t:
    f.write(str(i) + '\n')
f.close()

#y_pred_or = y_pred_or.reshape(146001)
#y_actual = pd.read_csv('data/test_pred.csv').to_numpy()
#y_actual = y_actual.reshape(rows)

#print(erro(y_pred_t, y_actual))
#print(erro(y_pred_or, y_actual))
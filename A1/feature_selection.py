import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoLars



train_file = sys.argv[1]
created_file = sys.argv[2]
selected_file = sys.argv[3]
target_cols = ['Zip Code - 3 digits', 'Age Group', 'CCSR Diagnosis Description', 'APR DRG Description', 'APR MDC Description','CCSR Procedure Description', 'Permanent Facility Id']
encoder_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
drop = ['CCSR Diagnosis Code', 'CCSR Procedure Code', 'APR DRG Code', 'APR MDC Code', 'Operating Certificate Number', 'Facility Name']
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

f = open(created_file, "w")
for col in df_trans.columns:
    f.write(col + '\n')
for col in drop:
    f.write(col + '\n')
f.close()

f = open(selected_file, "w")
for col in df_trans.columns:
    f.write('1' + '\n')
for col in drop:
    f.write('0' + '\n')
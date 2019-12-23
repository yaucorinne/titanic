# %%
# Author: Corinne Yau


# %%
def get_feat_vector(df,mode='train',train_columns=None):
    '''
    Returns Passenger ID's and corrensponding feature vector with relevant one-hot vector encoding,
    if mode is set to 'train', train_columns is returned which should be inputted for 'test' mode.
    '''
    features=['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
    passenger_ids=df['PassengerId']
    df_out=df[features]

    #converting cabin number to cabin code (first letter only)
    df_out['Cabin']=df_out['Cabin'].str[:1]

    df_out=pd.get_dummies(df_out,columns=['Sex','Cabin','Embarked'])

    if mode=='train':
    
        #check for nans
        for c in df_out.columns:
            has_nan=np.isnan(df_out[c]).any()
            if has_nan==True:
                df_out[c+'_nan']=np.where(np.isnan(df_out[c]),1,0)
                df_out[c]=np.where(np.isnan(df_out[c]),0,df_out[c])

        train_columns=df_out.columns
        return passenger_ids, df_out, train_columns
    
    elif mode=='test':

        #check for nans
        for c in df_out.columns:
            has_nan=np.isnan(df_out[c]).any()
            if has_nan==True:
                df_out[c+'_nan']=np.where(np.isnan(df_out[c]),1,0)
                df_out[c]=np.where(np.isnan(df_out[c]),0,df_out[c])

        test_columns=df_out.columns
        columns_to_delete=list(set(test_columns)-set(train_columns))
        columns_to_add=list(set(train_columns)-set(test_columns))
        if columns_to_delete:
            df_out.drop(columns=columns_to_delete,inplace=True)
        if columns_to_add:
            for i in columns_to_add:
                df_out.insert(0,i,0)
   
        df_out=df_out[train_columns]

        return passenger_ids, df_out

    else:
        raise Exception("The 'mode' argument should be set to either 'train' or 'test' ")

# %%

import sys
print('SYSPATH1: ',sys.path)
'''
import os
cwd=os.getcwd()
v_cwd=cwd+'/venv/lib/'
v_paths=[cwd,v_cwd+'python36.zip',v_cwd+'python3.6',v_cwd+'python3.6/lib-dynload',
v_cwd+'python3.6/site-packages']
sys.path=sys.path+v_paths

print('SYSPATH2: ',sys.path)
'''
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb



# %%

filepath=r'/Users/corinneyau/Documents/TitanicML/titanic/data/'
train_file='train.csv'
test_file='test.csv'

df_train=pd.read_csv(filepath+train_file)
df_test= pd.read_csv(filepath+test_file)

df_train_Y= df_train['Survived']
df_train_X=df_train.drop(columns='Survived')

# %%
train_ids, train_feat_vector, train_columns=get_feat_vector(df_train_X,mode='train')
train_labels=df_train_Y

test_ids, test_feat_vector=get_feat_vector(df_test,mode='test',train_columns=train_columns)

# %%

k_splitter=StratifiedKFold(5)
for train_index, val_index in k_splitter.split(train_feat_vector,train_labels):
    train_vector=train_feat_vector.iloc[train_index]
    train_ans=train_labels.iloc[train_index]
    val_vector=train_feat_vector.iloc[val_index]
    val_ans=train_labels.iloc[val_index]

    dtrain = xgb.DMatrix(train_vector, label=train_ans)
    dval  = xgb.DMatrix(val_vector,label=val_ans)

    ### Nee
    param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = ['auc','error','rmse']

    evallist = [(dval, 'eval'), (dtrain, 'train')]

    num_round = 10
    bst = xgb.train(param, dtrain, num_round,evallist)


print('CODE SOMEHOW GOT TO END')


# %%

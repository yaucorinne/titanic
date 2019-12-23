# %%


def get_feat_vector(df,mode='train',train_columns=None):
    '''
    Returns Passenger ID's and corrensponding feature vector with relevant one-hot vector encoding
    '''
    features=['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
    passenger_ids=df['PassengerId']
    df_out=df[features]

    #converting cabin number to cabin code (first letter only)
    df_out['Cabin']=df_out['Cabin'].str[:1]

    df_out=pd.get_dummies(df_out,columns=['Sex','Cabin','Embarked'])

    if mode=='train':
        train_columns=df_out.columns

        return passenger_ids, df_out, train_columns
    
    elif mode=='test':
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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
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
test_ids, test_feat_vector=get_feat_vector(df_test,mode='test',train_columns=train_columns)

# %%

# %%

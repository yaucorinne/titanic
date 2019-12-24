import pandas as pd
import numpy as np

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

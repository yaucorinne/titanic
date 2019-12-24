import pandas as pd
from feat_vector_func import get_feat_vector


def load_data():

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
    return train_ids, train_feat_vector, train_labels, test_ids, test_feat_vector
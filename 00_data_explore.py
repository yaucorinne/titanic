import pandas as pd
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

features=df_train_X.keys()

print('pause')

# %%


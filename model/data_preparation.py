
import pandas as pd
import numpy as np
from sklearn import preprocessing

df_train = pd.read_csv("../application_train.csv")

#remplacer les string par des int (tokenize)
print('Replacing string values by integers')
for i in df_train.columns:
    print(i)
    l=[]
    for y in df_train[i].index:
        value=df_train[i][y]
        if isinstance(value,str):
            if value not in l:
                l.append(value)
                print(value)
                print(l.index(value)) 
            df_train[i] = df_train[i].replace([value],l.index(value))       
        else : continue

#Supprime les colonnes qui ont trop de NaN (un peu plus de 40%)
print('Deleting columns containing too many NaNs')
for i in df_train.isnull().sum().index :
    if df_train[i].isnull().sum() > df_train.index.size/2.3:
        print(df_train[i])
        df_train=df_train.drop(i,1)

#supprime les lignes qui ont plus de 9 NaN
print('Deleting rows containing too many NaNs')
df_essai=df_train.isnull().sum(axis=1)
print(df_train.isnull().sum(axis=1))
s=0
for i in df_essai.index :
    if df_essai[i]> 8:
        df_essai=df_essai.drop(i,0)
        df_train=df_train.drop(i,0)
        print('Row',i,'deleted')

#transforming remaining NaNs to median of column
print('transforming remaining NaNs to median of column')
for i in df_train.columns : 
    df_train[i]=df_train[i].fillna(df_train[i].median())


df_train.to_csv("../prepared_application_train.csv",index=False)










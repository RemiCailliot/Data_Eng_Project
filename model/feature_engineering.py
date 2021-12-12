
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import normalize 
import io

df_train=pd.read_csv("../prepared_application_train.csv")
#df_train.columns

#removing outliers
Q1 = df_train.quantile(0.05)
Q3 = df_train.quantile(0.95)
IQR = Q3 - Q1

df_train = df_train[~((df_train < (Q1 - 1.5 * IQR)) |(df_train > (Q3 + 1.5 * IQR))).any(axis=1)]

print(df_train)

#normalize

min_max_scaler = preprocessing.MinMaxScaler()
df_train_scaled= min_max_scaler.fit_transform(df_train)
df_train_normalized=pd.DataFrame(df_train_scaled,columns=df_train.columns)

df_train_normalized.to_csv("../final_application_train.csv",index=False)

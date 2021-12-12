
import pandas as pd
import numpy as np


df = pd.read_csv('../data/Twitter_Data.csv')
df=df.dropna()
df.to_csv("../data/prepared_application_train.csv",index=False)










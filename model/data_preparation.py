
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

df = pd.read_csv('./data/Twitter_Data.csv')
print("Dropping NaN value in the dataset...")
df=df.dropna()
print("Saving dataset...")
df.to_csv("./data/prepared_application_train.csv",index=False)
print("Done!")










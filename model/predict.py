from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
print("Starting predictions...")
import mlflow
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

value="Hello Guys i'm fine"
model = pickle.load(open('./predict_src/clf.sav', 'rb'))

# with open('./predict_src/alpha.txt', "r") as myfile:
#     alpha = myfile.readlines()
# with open('./predict_src/l1_ratio.txt', "r") as myfile:
#     l1_ratio = myfile.readlines()

# alpha = float(alpha[0])
# l1_ratio = float(l1_ratio[0])

# print('Alpha :', alpha)
# print('Learning rate :', l1_ratio)

    #predictions
predicted= model.predict(value)
print("value:",value)


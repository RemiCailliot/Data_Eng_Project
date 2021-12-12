from sklearn.ensemble import GradientBoostingClassifier
import logging
import os
import warnings
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost
import shap
import matplotlib.pylab as pl
from sklearn.linear_model import ElasticNet
import pickle
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
shap.initjs()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



warnings.filterwarnings("ignore")
np.random.seed(40)

df=pd.read_csv("../final_application_train.csv")
df.drop(["SK_ID_CURR"],axis=1)
train, test = train_test_split(df,test_size=0.25, random_state=42)


X_train=train.drop(["TARGET"],axis=1)
X_test=test.drop(["TARGET"],axis=1)
y_train=train[["TARGET"]]
y_test=test[["TARGET"]]


alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

#Xgboost model
model = XGBClassifier(eta=l1_ratio, alpha=alpha)
model.fit(X_train, y_train)
#Random Forest model
model2 = RandomForestClassifier(
    n_estimators=500, bootstrap=True, max_features='sqrt')
model2.fit(X_train, y_train)
#Gradient Boosting model
model3 = GradientBoostingClassifier(
    n_estimators=100, learning_rate=l1_ratio, max_depth=1, random_state=0)
model3.fit(X_train, y_train)

X_test.to_csv("./predict_src/X_test.csv", index=False)
y_test.to_csv("./predict_src/y_test.csv", index=False)
pickle.dump(model, open('./predict_src/model.sav', 'wb'))
pickle.dump(model2, open('./predict_src/model2.sav', 'wb'))
pickle.dump(model3, open('./predict_src/model3.sav', 'wb'))
with open('./predict_src/alpha.txt', 'w') as Alpha:
    Alpha.write(str(alpha))
with open('./predict_src/l1_ratio.txt', 'w') as L1_ratio:
    L1_ratio.write(str(l1_ratio))

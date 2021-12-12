from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
print("Starting predictions...")
from urllib.parse import urlparse
import mlflow
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

X_test = pd.read_csv('./predict_src/X_test.csv')
y_test = pd.read_csv('./predict_src/y_test.csv')
model = pickle.load(open('./predict_src/model.sav', 'rb'))
model2 = pickle.load(open('./predict_src/model2.sav', 'rb'))
model3 = pickle.load(open('./predict_src/model3.sav', 'rb'))
with open('./predict_src/alpha.txt', "r") as myfile:
    alpha = myfile.readlines()
with open('./predict_src/l1_ratio.txt', "r") as myfile:
    l1_ratio = myfile.readlines()

alpha = float(alpha[0])
l1_ratio = float(l1_ratio[0])

print('Alpha :', alpha)
print('Learning rate :', l1_ratio)
# # Useful for multiple runs (only doing one run in this sample notebook)
with mlflow.start_run():

   #predictions
    rf_predictions = model2.predict(X_test)
    rf_probs = model2.predict_proba(X_test)[:, 1]
    roc_value = roc_auc_score(y_test, rf_probs)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("XGBoost accuracy: %.2f%%" % (accuracy * 100.0))
    print("Random Forest accuracy: %.2f%%" % (roc_value * 100.0))
    print("Gradient Boosting accuracy: %.2f%%" %
          (model3.score(X_test, y_test) * 100))

    # Evaluate Metrics
    (rmse, mae, r2) = eval_metrics(y_test, predictions)

    # Print out metrics
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # Log parameter, metrics, and model to MLflow
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("accuracy xgboost", accuracy*100.0)
    mlflow.log_metric("accuracy random forest", roc_value*100.0)
    mlflow.log_metric("accuracy gradient boosting",
                      model3.score(X_test, y_test)*100)
tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

# Model registry does not work with file store
if tracking_url_type_store != "file":

    # Register the model
    # There are other ways to use the Model Registry, which depends on the use case,
    # please refer to the doc for more information:
    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
    mlflow.sklearn.log_model(
        model, "model", registered_model_name="ElasticnetWineModel")
else:
    mlflow.sklearn.log_model(model, "model")

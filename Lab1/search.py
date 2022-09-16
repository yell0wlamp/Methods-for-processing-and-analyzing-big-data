import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import cv, XGBClassifier


def labelEncoder(column):
    le = LabelEncoder().fit(list(train[column].values) + list(test[column].values))
    train[column] = le.transform(train[column].values)
    test[column] = le.transform(test[column].values)


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
features = ['Gender', 'Pensioner', 'Married', 'Children', 'Months_with_company',
            'Months_with_company', 'Phone_Service', 'Multiple_Lines', 'Internet_Service',
            'Online_Security', 'Online_Backup', 'Device_Protection', 'Tech_Support',
            'Streaming_TV', 'Streaming_Movies', 'Type_contract', 'Paperless_Billing',
            'Payment_Method', 'Monthly_Payment', 'Total_Payment']

for feature in features:
    labelEncoder(feature)

x = train.drop('Leave_Next_Month', axis=1)
y = train['Leave_Next_Month']
data_dmatrix = xgb.DMatrix(data=x, label=y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
params = {
    "objective": "binary:logistic",
    "eval_metric": 'logloss',
    "eta": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 1,
}
xgb_clf = XGBClassifier(**params)
xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)
print('XGBoost model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=4, num_boost_round=50, early_stopping_rounds=10,
            metrics="auc", seed=123)


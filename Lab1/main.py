import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score


def labelEncoder(column):
    le = LabelEncoder().fit(list(train[column].values) + list(test[column].values))
    train[column] = le.transform(train[column].values)
    test[column] = le.transform(test[column].values)


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# sample = pd.read_csv('sample_submission.csv')
features = ['Gender', 'Pensioner', 'Married', 'Children', 'Months_with_company',
            'Months_with_company', 'Phone_Service', 'Multiple_Lines', 'Internet_Service',
            'Online_Security', 'Online_Backup', 'Device_Protection', 'Tech_Support',
            'Streaming_TV', 'Streaming_Movies', 'Type_contract', 'Paperless_Billing',
            'Payment_Method', 'Monthly_Payment', 'Total_Payment']

for feature in features:
    labelEncoder(feature)

if __name__ == '__main__':
    params_xgb = {
        "objective": "binary:logistic",
        "eval_metric": 'logloss',
        "eta": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 1,
    }

    num_boost_round = 100
    early_stopping_rounds = 10

    dtrain = xgb.DMatrix(train[features].values, train['Leave_Next_Month'].values)
    dvalid = xgb.DMatrix(train[features].values, train['Leave_Next_Month'].values)

    watchlist = [(dtrain, 'Leave_Next_Month'), (dvalid, 'eval')]
    gbm = xgb.train(params_xgb, dtrain, num_boost_round, evals=watchlist,
                    early_stopping_rounds=early_stopping_rounds, verbose_eval=10)

    pred = gbm.predict(xgb.DMatrix(test[features].values), ntree_limit=gbm.best_iteration + 1)
    # pred = np.round(pred)

    sample = {'id': test['id'],
              'Leave_Next_Month': pred
              }
    sample_df = pd.DataFrame(sample)
    sample_df.to_csv('My.csv', index=False)
    # accuracy = accuracy_score(train['Leave_Next_Month'].values, np.round(pred))
    # auc = roc_auc_score(train['Leave_Next_Month'].values, pred)
    # print('Accuracy: {:.2f} %, ROC AUC: {:.2f}'.format(100 * accuracy, auc))

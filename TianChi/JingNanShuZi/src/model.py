from sklearn.model_selection import KFold
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error


def lgbModel(train_length,test_length,X_train,Y_train,X_test):
    lgb_param = {'num_leaves': 31,
                 'min_data_in_leaf': 30,
                 'objective':'regression',
                 'max_depth': -1,
                 'learning_rate': 0.01,
                 "min_child_samples": 30,
                 "boosting": "gbdt",
                 "feature_fraction": 0.9,
                 "bagging_freq": 1,
                 "bagging_fraction": 0.9 ,
                 "bagging_seed": 12,
                 "metric": 'mse',
                 'num_threads':8,
                  "lambda_l1": 0.1,
                 "lambda_l2": 0.2,
                 "verbosity": -1}

    folds = KFold(n_splits=5, shuffle=True, random_state=2017)
    oof_lgb = np.zeros(train_length)
    predictions_lgb = np.zeros(test_length)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, Y_train)):
        print("fold n°{}".format(fold_+1))
        trn_data = lgb.Dataset(X_train[trn_idx], Y_train[trn_idx])
        val_data = lgb.Dataset(X_train[val_idx], Y_train[val_idx])

        num_round = 10000
        clf = lgb.train(lgb_param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 200)
        oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)

        predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

    print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, Y_train)))
    return predictions_lgb,oof_lgb

import xgboost as xgb


def xgbModel(train_length,test_length,X_train,Y_train,X_test):

    xgb_params = {'eta': 0.005, 'max_depth': 4 ,'subsample': 0.8, 'colsample_bytree': 0.7,
              'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 8}

    folds = KFold(n_splits=5, shuffle=True, random_state=2017)
    oof_xgb = np.zeros(train_length)
    predictions_xgb = np.zeros(test_length)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, Y_train)):
        print("fold n°{}".format(fold_+1))
        trn_data = xgb.DMatrix(X_train[trn_idx], Y_train[trn_idx])
        val_data = xgb.DMatrix(X_train[val_idx], Y_train[val_idx])

        watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
        clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=400, params=xgb_params)
        oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
        predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

    print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, Y_train)))
    return predictions_xgb,oof_xgb

from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.linear_model import BayesianRidge
from sklearn import linear_model


def modelResultMerge(predictions_lgb,predictions_xgb,train_lgb,train_xgb,target):
    train_stack = np.vstack([train_lgb, train_xgb]).transpose()
    test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

    folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2012)
    oof_stack = np.zeros(train_stack.shape[0])
    predictions = np.zeros(test_stack.shape[0])

    for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, target)):
        print("fold {}".format(fold_))
        trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
        val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

        clf_3 = BayesianRidge()
        clf_3.fit(trn_data, trn_y)

        #     clf_3 = linear_model.LinearRegression()
        #     clf_3.fit(trn_data, trn_y)

        oof_stack[val_idx] = clf_3.predict(val_data)
        predictions += clf_3.predict(test_stack) / 10

    print(mean_squared_error(target.values, oof_stack))
    return predictions

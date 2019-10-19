import numpy as np
import matplotlib.pyplot as plt
import sklearn.feature_extraction.text
import sklearn.grid_search
import sklearn.linear_model
import sklearn.neighbors
import sklearn.ensemble
import xgboost as xgb
import sklearn.decomposition

import input_output as io
import feature_extraction as fe
import metafeatures as mf
import tuning as tuning
import word2vec_model as w2v
import tfidf as tfidf


import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def watch_errors(X, target):
    X = X.values
    preds = np.zeros(target.shape[0], dtype=np.float32)
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=4, subsample=0.9,
                             colsample_bytree=1.0, min_child_weight=2, seed=0)

    kf = sklearn.cross_validation.KFold(X.shape[0], n_folds=5,
                                        shuffle=True, random_state=1234)

    for train_index, test_index in kf:
        model.fit(X[train_index], target[train_index])
        preds[test_index] = model.predict(X[test_index])


    index = np.argsort(np.abs(preds - target))
    preds = preds[index]
    order = np.arange(len(X))[index]

    return preds, order


def make_predictions(train, target, test):
    result = np.zeros(len(test))
    seed_list = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 123]
    for seed in seed_list:
        model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.06, max_depth=8, subsample=0.9,
                                 colsample_bytree=0.4, min_child_weight=16, seed=seed)
        model.fit(train, target)
        result += model.predict(test)

    result /= len(seed_list)
    return result


def run_w2v(train, test):
    w2v.make_word2vec_model(train, test)
    w2v.word2vec_features(train, test, load=False)


def run_tfidf(train, test):
    tfidf.make_tfidf_model(train, test)


def preds_train(X, target):
    X = X.values
    preds = np.zeros(len(target))
    cv = sklearn.cross_validation.StratifiedKFold(target, n_folds=5, shuffle=True, random_state=1234)

    for train_index, test_index in cv:
        model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.06, max_depth=8, subsample=0.9,
                                 colsample_bytree=0.4, min_child_weight=16, seed=0)
        model.fit(X[train_index], target[train_index])
        preds[test_index] = model.predict(X[test_index])

    return preds


def add_metafeature(train_, target_, test_):
    target = np.round(target_ * 100)
    metafeature_train = np.zeros((len(target), len(np.unique(target))))
    metafeature_test = np.zeros((len(test_), len(np.unique(target))))
    model = sklearn.linear_model.LogisticRegression(penalty="l2", C=1)

    train = train_.values
    test = test_.values

    kfold = sklearn.cross_validation.StratifiedKFold(target, 4)
    for train_index, test_index in kfold:
        model.fit(train[train_index], target[train_index])
        metafeature_train[test_index] = model.predict_proba(train[test_index])
        metafeature_test += model.predict_proba(test)/4

    return metafeature_train, metafeature_test

try:
    del train
    del test
except:
    print("First run")


train, test, target, attributes, test_index = io.load_data()
train, test = fe.preprocess_data(train, test, attributes, load=True)
#0.463342320579

#run_w2v(train, test)
#run_tfidf(train, test)
"""
train_tfidf, test_tfidf = fe.generate_tfidf(train, test)
tuning.parametr_tuning(train_tfidf, target)
"""
train, test = fe.extract_features(train), fe.extract_features(test)
#mf_train, mf_test = add_metafeature(train, target, test)
#train = np.column_stack((train.values, mf_train))
#test = np.column_stack((test.values, mf_test))
#preds, order = watch_errors(train, target)
#preds = preds_train(train, target)
tuning.parametr_tuning(train, target, transform_target="None")
"""
result = make_predictions(train, target, test)
io.save_result(test_index, result)
"""
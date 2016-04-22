import numpy as np
import sklearn.feature_extraction.text
import sklearn.grid_search
import sklearn.linear_model
import sklearn.manifold
import xgboost as xgb
import pickle
import scipy
import sklearn.calibration
from sklearn.externals import joblib

import input_output as io
import feature_extraction as fe
import tuning as tuning


def make_predictions(load_list=[]):
    
    train, test, target, test_index = io.load_data()
    result = np.zeros(len(test))
    seed_list = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 123]
    mcw_list = [1, 1, 1, 1, 1, 8, 8, 8, 8, 8]
    seed_list2 = [1242, 5432]
    mcw_list2 = [1, 8]
    
    train, test, target, test_index = io.load_data()
    train, test, target = fe.preprocess_data(train, test, target, preprocess_type=1)
    train_meta, test_meta = train_level1(load_list=["xgb1", "xgb2", "knn", "ext1", "ext2", "rf1", "rf2"])
    train = np.column_stack((train, train_meta))
    test = np.column_stack((test, test_meta))
    
    print "Training model1..."
    pred1 = np.zeros(len(test))
    for i in range(10):
        if "xgb1" not in load_list:
            model1 = xgb.XGBClassifier(n_estimators=550, learning_rate=0.01, max_depth=6, colsample_bytree=0.95,
                                       subsample=1, min_child_weight=mcw_list[i], seed=seed_list[i])
            model1.fit(train, target)
            pickle.dump(model1, open("final_models/xgb/1/xgb_n_"+str(i)+".pkl", "wb"))
        else:
            model1 = pickle.load(open("final_models/xgb/1/xgb_n_"+str(i)+".pkl", "rb"))
        pred1 += model1.predict_proba(test)[:, 1]
    pred1 /= len(seed_list)

    print "Training model2..."
    pred2 = np.zeros(len(test))
    for i in range(2):
        if "xgb2" not in load_list:
            model2 = xgb.XGBClassifier(n_estimators=550, learning_rate=0.01, max_depth=6, colsample_bytree=0.95,
                                       subsample=1, min_child_weight=mcw_list2[i], seed=seed_list2[i])
            model2 = sklearn.calibration.CalibratedClassifierCV(model2, method="isotonic", cv=10)
            model2.fit(train, target)
            pickle.dump(model2, open("final_models/xgb/2/xgb_n_"+str(i)+".pkl", "wb"))
        else:
            model2 = pickle.load(open("final_models/xgb/2/xgb_n_"+str(i)+".pkl", "rb"))
        pred2 += model2.predict_proba(test)[:, 1]
    pred2 /= len(seed_list2)

    print "Training model4..."
    pred4 = np.zeros(len(test))
    if "ext1" not in load_list:
        model4 = sklearn.ensemble.ExtraTreesClassifier(n_estimators=1000,max_features=50,criterion='entropy',min_samples_split=4,
                                                       max_depth=35, min_samples_leaf=2, n_jobs =-1, random_state=1234)
        model4.fit(train, target)
        pred4 = model4.predict_proba(test)[:, 1]        
        pred4.dump("final_models/ext/ext1_pred")
    else:
        pred4 = np.load("final_models/ext/ext1_pred")
            
    result = 0.7*np.sqrt(pred1*pred2) + 0.3*pred4
    
    return result

def train_level1(load_list=[]):    
    print "Training xgb1..."
    train, test, target, test_index = io.load_data()
    train, test, target = fe.preprocess_data(train, test, target, preprocess_type=1)

    N_FOLDS = 10
    cv = sklearn.cross_validation.StratifiedKFold(target, n_folds=N_FOLDS, shuffle=True, random_state=1234)
    train_xgb1 = np.zeros(train.shape[0])
    test_xgb1 = np.zeros(test.shape[0])
    for fold, (train_index, test_index) in enumerate(cv):
        if "xgb1" not in load_list:
            model = xgb.XGBClassifier(n_estimators=180, learning_rate=0.05, max_depth=11, colsample_bytree=0.8,
                                      subsample=0.96, min_child_weight=4, seed=1234)
            model.fit(train[train_index], target[train_index])
            pickle.dump(model, open("level1/xgb/1/xgb_1_fold_"+str(fold)+".pkl", "wb"))
        else:
            model = pickle.load(open("level1/xgb/1/xgb_1_fold_"+str(fold)+".pkl", "rb"))
        train_xgb1[test_index] = model.predict_proba(train[test_index])[:, 1]
        test_xgb1 += model.predict_proba(test)[:, 1]/N_FOLDS

    train_meta = train_xgb1.reshape((train_xgb1.shape[0], 1))
    test_meta = test_xgb1.reshape((test_xgb1.shape[0], 1)) 
    
    print "Training xgb2..."
    train, test, target, test_index = io.load_data()
    train, test, target = fe.preprocess_data(train, test, target, preprocess_type=1)

    N_FOLDS = 10
    cv = sklearn.cross_validation.StratifiedKFold(target, n_folds=N_FOLDS, shuffle=True, random_state=1234)
    train_xgb2 = np.zeros(train.shape[0])
    test_xgb2 = np.zeros(test.shape[0])
    for fold, (train_index, test_index) in enumerate(cv):
        if "xgb2" not in load_list:
            model = xgb.XGBRegressor(n_estimators=600, learning_rate=0.02, max_depth=9, colsample_bytree=1,
                                     subsample=1, min_child_weight=1, seed=1234)
            model.fit(train[train_index], target[train_index])
            pickle.dump(model, open("level1/xgb/2/xgb_1_fold_"+str(fold)+".pkl", "wb"))
        else:
            model = pickle.load(open("level1/xgb/2/xgb_1_fold_"+str(fold)+".pkl", "rb"))
        train_xgb2[test_index] = model.predict(train[test_index])
        test_xgb2 += model.predict(test)/N_FOLDS

    train_meta = np.column_stack((train_meta, train_xgb2))
    test_meta = np.column_stack((test_meta, test_xgb2))

    
    print "Training knn..."
    if "knn" not in load_list:
        train, test, target, test_index = io.load_data()
        train, test, target = fe.preprocess_data(train, test, target, preprocess_type=4)
    
        N_FOLDS = 10
        cv = sklearn.cross_validation.StratifiedKFold(target, n_folds=N_FOLDS, shuffle=True, random_state=1234)
        train_knn = np.zeros(train.shape[0])
        test_knn = np.zeros(test.shape[0])
        for fold, (train_index, test_index) in enumerate(cv):
            model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=128, metric="minkowski", weights="distance", n_jobs=-1)
            model.fit(train[train_index], target[train_index])
            train_knn[test_index] = model.predict_proba(train[test_index])[:, 1]
        model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=128, metric="minkowski", weights="distance", n_jobs=-1)
        model.fit(train, target)
        test_knn = model.predict_proba(test)[:, 1]
        train_knn.dump("level1/knn/train_knn")
        test_knn.dump("level1/knn/test_knn")
    else:
        train_knn = np.load("level1/knn/train_knn")
        test_knn = np.load("level1/knn/test_knn")
     
    train_meta = np.column_stack((train_meta, train_knn))
    test_meta = np.column_stack((test_meta, test_knn))
    
    print "Training ext1..."
    if "ext1" not in load_list:
        train, test, target, test_index = io.load_data()
        train, test, target = fe.preprocess_data(train, test, target, preprocess_type=3)
    
        N_FOLDS = 10
        cv = sklearn.cross_validation.StratifiedKFold(target, n_folds=N_FOLDS, shuffle=True, random_state=1234)
        train_ext1 = np.zeros(train.shape[0])
        test_ext1 = np.zeros(test.shape[0])
        for fold, (train_index, test_index) in enumerate(cv):
            model = sklearn.ensemble.ExtraTreesClassifier(n_estimators=1000,max_features=50,criterion='entropy',min_samples_split=4,
                                                       max_depth=35, min_samples_leaf=2, n_jobs =-1, random_state=1234)
            model.fit(train[train_index], target[train_index])
            train_ext1[test_index] = model.predict_proba(train[test_index])[:, 1]
            test_ext1 += model.predict_proba(test)[:, 1]/N_FOLDS
        train_ext1.dump("level1/ext/1/train_ext")
        test_ext1.dump("level1/ext/1/test_ext")
    else:
        train_ext1 = np.load("level1/ext/1/train_ext")
        test_ext1 = np.load("level1/ext/1/test_ext")

    train_meta = np.column_stack((train_meta, train_ext1))
    test_meta = np.column_stack((test_meta, test_ext1))

    
    print "Training ext2..."
    if "ext2" not in load_list:
        train, test, target, test_index = io.load_data()
        train, test, target = fe.preprocess_data(train, test, target, preprocess_type=5)
    
        N_FOLDS = 10
        cv = sklearn.cross_validation.StratifiedKFold(target, n_folds=N_FOLDS, shuffle=True, random_state=1234)
        train_ext2 = np.zeros(train.shape[0])
        test_ext2 = np.zeros(test.shape[0])
        for fold, (train_index, test_index) in enumerate(cv):
            model = sklearn.ensemble.ExtraTreesRegressor(n_estimators=1000,max_features=50,min_samples_split=4,
                                                         max_depth=35, min_samples_leaf=2, n_jobs =-1, random_state=1234)
            model.fit(train[train_index], target[train_index])
            train_ext2[test_index] = model.predict(train[test_index])
            test_ext2 += model.predict(test)/N_FOLDS
        train_ext2.dump("level1/ext/2/train_ext")
        test_ext2.dump("level1/ext/2/test_ext")
    else:
        train_ext2 = np.load("level1/ext/2/train_ext")
        test_ext2 = np.load("level1/ext/2/test_ext")
    
    train_meta = np.column_stack((train_meta, train_ext2))    
    test_meta = np.column_stack((test_meta, test_ext2))
    
    
    print "Training rf1..."
    if "rf1" not in load_list:
        train, test, target, test_index = io.load_data()
        train, test, target = fe.preprocess_data(train, test, target, preprocess_type=1)
    
        N_FOLDS = 10
        cv = sklearn.cross_validation.StratifiedKFold(target, n_folds=N_FOLDS, shuffle=True, random_state=1234)
        train_rf1 = np.zeros(train.shape[0])
        test_rf1 = np.zeros(test.shape[0])
        for fold, (train_index, test_index) in enumerate(cv):
            model = sklearn.ensemble.RandomForestClassifier(n_estimators=500, criterion="entropy", max_depth=32, min_samples_leaf=4, n_jobs=-1, random_state=1234)
            model.fit(train[train_index], target[train_index])
            train_rf1[test_index] = model.predict_proba(train[test_index])[:, 1]
            test_rf1 += model.predict_proba(test)[:, 1]/N_FOLDS
        train_rf1.dump("level1/rf/1/train_rf")
        test_rf1.dump("level1/rf/1/test_rf")
    else:
        train_rf1 = np.load("level1/rf/1/train_rf")
        test_rf1 = np.load("level1/rf/1/test_rf")

    train_meta = np.column_stack((train_meta, train_rf1))
    test_meta = np.column_stack((test_meta, test_rf1))

    
    print "Training rf2..."
    if "rf2" not in load_list:
        train, test, target, test_index = io.load_data()
        train, test, target = fe.preprocess_data(train, test, target, preprocess_type=1)
    
        N_FOLDS = 10
        cv = sklearn.cross_validation.StratifiedKFold(target, n_folds=N_FOLDS, shuffle=True, random_state=1234)
        train_rf2 = np.zeros(train.shape[0])
        test_rf2 = np.zeros(test.shape[0])
        for fold, (train_index, test_index) in enumerate(cv):
            model = sklearn.ensemble.RandomForestRegressor(n_estimators=500, max_depth=32, min_samples_leaf=4, n_jobs=-1, random_state=1234)
            model.fit(train[train_index], target[train_index])
            train_rf2[test_index] = model.predict(train[test_index])
            test_rf2 += model.predict(test)/N_FOLDS
        train_rf2.dump("level1/rf/2/train_rf")
        test_rf2.dump("level1/rf/2/test_rf")
    else:
        train_rf2 = np.load("level1/rf/2/train_rf")
        test_rf2 = np.load("level1/rf/2/test_rf")    

    train_meta = np.column_stack((train_meta, train_rf2))
    test_meta = np.column_stack((test_meta, test_rf2))
    
    
    return train_meta, test_meta

   
def main():
    
    
    train, test, target, test_index = io.load_data()  
    train, test, target = fe.preprocess_data(train, test, target, preprocess_type=1)

    #tuning.parametr_tuning(train, target, param_grid={})
    #tuning.ensemble_tuning(train, test, target, load_list=["xgb1"])
    
    result = make_predictions(load_list=["xgb1", "xgb2", "ext1"])
    io.save_result(test_index, result)
    
    
if __name__ == "__main__":
    main()
import numpy as np
import sklearn.cross_validation
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
import sklearn.base
import sklearn.svm
import sklearn.naive_bayes
import sklearn.preprocessing
import xgboost as xgb
import pickle
import sklearn.calibration

import feature_extraction as fe
import input_output as io
import main as main


def ensemble_tuning(train, test, target, load_list=[]):
    N_FOLDS = 10
    seed_list = [1234, 2345, 6789, 7890]
    mcw_list = [1, 1, 8, 8]
    
    cv = sklearn.cross_validation.StratifiedKFold(target, n_folds=N_FOLDS, shuffle=True, random_state=1234)
    scores = np.zeros((N_FOLDS, 101), dtype=np.float32)

    train, test, target, test_index = io.load_data()
    X1, _, target = fe.preprocess_data(train, test, target, preprocess_type=1)
    train_meta, test_meta = main.train_level1(load_list=["xgb1", "xgb2", "knn", "ext1", "ext2", "rf1", "rf2"])
    X1 = np.column_stack((X1, train_meta))
    
    for fold, (train_index, test_index) in enumerate(cv):

        print("Training model1...")
        pred1 = np.zeros(len(X1[test_index]))
                
        for i in range(4):
            if "xgb1" not in load_list:
                model1 = xgb.XGBClassifier(n_estimators=550, learning_rate=0.01, max_depth=6, colsample_bytree=0.95,
                                       subsample=1, min_child_weight=mcw_list[i], seed=seed_list[i])
                model1.fit(X1[train_index], target[train_index])
                pickle.dump(model1, open("cv/xgb/1/xgb_1_fold_"+str(fold)+"_n_"+str(i)+".pkl", "wb"))
            else:
                model1 = pickle.load(open("cv/xgb/1/xgb_1_fold_"+str(fold)+"_n_"+str(i)+".pkl", "rb"))
            pred1 += model1.predict_proba(X1[test_index])[:, 1]
        pred1 /= len(seed_list)

        #print("Training model2...")
        #pred2 = np.zeros(len(X1[test_index]))
        
        """
        for i in range(6):
            if "xgb2" not in load_list:
                model2 = xgb.XGBRegressor(n_estimators=550, learning_rate=0.01, max_depth=6, colsample_bytree=0.95,
                                          subsample=1, min_child_weight=mcw_list[i], seed=seed_list[i])
                model2.fit(X1[train_index], target[train_index])
                pickle.dump(model1, open("cv/xgb/2/xgb_1_fold_"+str(fold)+"_n_"+str(i)+".pkl", "wb"))
            else:
                model2 = pickle.load(open("cv/xgb/2/xgb_1_fold_"+str(fold)+"_n_"+str(i)+".pkl", "rb"))
            pred2 += model1.predict(X1[test_index])
        pred2 /= len(seed_list)
        
        pred2[pred2 >= 0.99] = 0.99
        pred2[pred2 <= 0.01] = 0.01
        """
        
        print("Training model3...")
        if "ext1" not in load_list:
            model3 = sklearn.ensemble.ExtraTreesClassifier(n_estimators=1000,max_features=50,criterion='entropy',min_samples_split=4,
                                                           max_depth=35, min_samples_leaf=2, n_jobs =-1, random_state=1234)
            model3.fit(X1[train_index], target[train_index])
            pred3 = model3.predict_proba(X1[test_index])[:, 1]
            pred3.dump("cv/rf/pred_fold_"+str(fold))
        else:
            pred3 = np.load("cv/rf/pred_fold_"+str(fold))
        
        print("Calculating scores...")
        for alpha in np.ndindex(101):
            scores[fold][alpha] = sklearn.metrics.log_loss(target[test_index], 0.01*alpha[0]*pred1 + np.max(1 - 0.01*alpha[0], 0)*pred3)#np.power(pred1**(0.01*alpha[0])*pred2**(0.01*alpha[1]), 1/(0.01*(alpha[0] + alpha[1] + 1))))
            #                                                    0.01*alpha[0]*pred1 + np.max(1 - 0.01*alpha[0], 0)*pred2)
        print("Current fold:", np.min(scores[fold]), np.unravel_index(scores[fold].argmin(), scores[fold].shape), scores[fold][100], scores[fold][0])
        sc1 = np.mean(scores, axis=0) * 1.0 / (fold+1) * N_FOLDS
        print("Accumulated:", np.min(sc1), np.unravel_index(sc1.argmin(), sc1.shape), sc1[100], sc1[0])

    scores1 = np.mean(scores, axis=0)
    print(np.min(scores1), np.unravel_index(scores1.argmin(), scores1.shape), scores1[100], scores1[0])

    return scores


def parametr_tuning(X, y, param_grid):
    #cv = sklearn.cross_validation.StratifiedKFold(y, n_folds=10, random_state=1234)
    cv = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=5, test_size=0.05, random_state=1234)
    #model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=11, colsample_bytree=1,
    #                          subsample=1, min_child_weight=1, seed=1234)
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, colsample_bytree=0.95,
                              subsample=1, min_child_weight=1, seed=1234)
    #model = sklearn.calibration.CalibratedClassifierCV(model, method="isotonic", cv=10)

    #model = sklearn.linear_model.SGDClassifier(loss="log", alpha=0.01, l1_ratio=0.0, n_iter=200, random_state=1234, n_jobs=-1)

    #model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=1, n_jobs=-1, random_state=1234)
    #model = sklearn.calibration.CalibratedClassifierCV(model, method="isotonic", cv=5)

    #model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=128, metric="minkowski", weights="distance", n_jobs=-1)

    #model = sklearn.ensemble.ExtraTreesClassifier(n_estimators=1000,max_features= 50,criterion= 'entropy',min_samples_split= 4,
    #                                              max_depth= 35, min_samples_leaf= 2, n_jobs = -1)


    gs = sklearn.grid_search.GridSearchCV(model, param_grid, scoring="log_loss", cv=cv, verbose=10,
                                          n_jobs=1)
    gs.fit(X, y)

    print("Best score is: ", gs.best_score_)
    print("Best parametrs:")

    best_params = gs.best_estimator_.get_params()

    for param_name in sorted(best_params.keys()):
        print(param_name, ":", best_params[param_name])
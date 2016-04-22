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
import scipy.stats
import pickle


def ensemble_tuning(X, target, load_list=[]):
    N_FOLDS = 30
    seed_list = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 123]
    cv = sklearn.cross_validation.StratifiedShuffleSplit(target, n_iter=N_FOLDS, test_size=0.1, random_state=1234)
    scores = np.zeros((N_FOLDS, 101), dtype=np.float32)

    for fold, (train_index, test_index) in enumerate(cv):
        print("Training model1...")
        """
        if "linear" not in load_list:
            model1 = sklearn.linear_model.SGDClassifier(loss="log", alpha=1, l1_ratio=0.0, n_iter=100, random_state=1234)
            model1.fit(X[train_index], target[train_index])
            pickle.dump(model1, open("level1/linear/lr_fold_"+str(fold)+".pkl", "wb"))
        else:
            model1 = pickle.load(open("level1/linear/lr_fold_"+str(fold)+".pkl", "rb"))
        """

        if "rf_entropy" not in load_list:
            model1 = sklearn.ensemble.RandomForestClassifier(n_estimators=2000, max_depth=8, criterion="entropy", bootstrap=False,
                                                             min_samples_leaf=4, min_samples_split=2, random_state=1234)
            model1.fit(X[train_index], target[train_index])
            pickle.dump(model1, open("level1/rf/rf_entropy_fold_"+str(fold)+".pkl", "wb"))
        else:
            model1 = pickle.load(open("level1/rf/rf_entropy_fold_"+str(fold)+".pkl", "rb"))
        pred1 = model1.predict_proba(X[test_index])[:, 1]


        print("Training model2...")
        pred2 = np.zeros(len(pred1))
        for i in range(10):
            if "xgb" not in load_list:
                model2 = xgb.XGBClassifier(n_estimators=100, max_depth=3, colsample_bytree=0.9, subsample=1,
                                          learning_rate=0.1, seed=seed_list[i])
                model2.fit(X[train_index], target[train_index])
                pickle.dump(model2, open("level1/xgb/xgb_fold_"+str(fold)+"_n_"+str(i)+".pkl", "wb"))
            else:
                model2 = pickle.load(open("level1/xgb/xgb_fold_"+str(fold)+"_n_"+str(i)+".pkl", "rb"))
            pred2 += model2.predict_proba(X[test_index])[:, 1]
        pred2 /= len(seed_list)


        #print("Calculating scores...")
        for alpha in np.ndindex(101):
            scores[fold][alpha] = sklearn.metrics.log_loss(target[test_index],
                                                                0.01*alpha[0]*pred1 + np.max(1 - 0.01*alpha[0], 0)*pred2)
        print("Current fold:", np.min(scores[fold]), np.unravel_index(scores[fold].argmin(), scores[fold].shape), scores[fold][100], scores[fold][0])
        sc1 = np.mean(scores, axis=0) * 1.0 / (fold+1) * N_FOLDS
        print("Accumulated:", np.min(sc1), np.unravel_index(sc1.argmin(), sc1.shape), sc1[100], sc1[0])

    scores1 = np.mean(scores, axis=0)
    print(np.min(scores1), np.unravel_index(scores1.argmin(), scores1.shape), scores1[100], scores1[0])

    return scores

def tune_xgboost(X, y, load_list=[]):
    N_FOLDS = 30
    seed_list = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 123]

    cv = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=N_FOLDS, test_size=0.1, random_state=1234)
    scores = np.zeros(N_FOLDS)

    for fold, (train_index, test_index) in enumerate(cv):

        pred = np.zeros(len(X[test_index]))
        for i in range(10):
            if "xgb" not in load_list:
                model = xgb.XGBClassifier(n_estimators=600, max_depth=4, colsample_bytree=0.9, subsample=1,
                                          learning_rate=0.01, seed=seed_list[i])
                model.fit(X[train_index], y[train_index])
                pickle.dump(model, open("tune/xgb/xgb_fold_"+str(fold)+"_n_"+str(i)+".pkl", "wb"))
            else:
                model = pickle.load(open("tune/xgb/xgb_fold_"+str(fold)+"_n_"+str(i)+".pkl", "rb"))
            print(sklearn.metrics.log_loss(y[test_index], model.predict_proba(X[test_index])[:, 1]))
            pred += model.predict_proba(X[test_index])[:, 1]
        pred /= len(seed_list)
        scores[fold] = sklearn.metrics.log_loss(y[test_index], pred)
        print("Fold:", fold, "Score:", scores[fold])

    print("Mean score:", np.mean(scores), "| Std score:", np.std(scores))




def parametr_tuning(X, y, param_grid):
    cv = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=30, test_size=0.1, random_state=1234)
    #model = xgb.XGBClassifier(n_estimators=100, max_depth=3, colsample_bytree=0.8, subsample=1,
    #                          learning_rate=0.1, seed=2345)

    #model = sklearn.linear_model.SGDClassifier(loss="log", alpha=1, l1_ratio=0.0, n_iter=100, random_state=1234)

    model = sklearn.ensemble.RandomForestClassifier(n_estimators=300, max_depth=8, criterion="entropy", bootstrap=False,
                                                    min_samples_leaf=4, min_samples_split=2, random_state=1234)

    #model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=32, weights="distance")

    #model = sklearn.ensemble.ExtraTreesClassifier(n_estimators=300, criterion="entropy", max_depth=8, bootstrap=False,
    #                                              min_samples_leaf=1, min_samples_split=2, random_state=1234)



    gs = sklearn.grid_search.GridSearchCV(model, param_grid, scoring="log_loss", cv=cv, verbose=10,
                                          n_jobs=1)
    gs.fit(X, y)

    print("Best score is: ", gs.best_score_)
    print("Best parametrs:")

    best_params = gs.best_estimator_.get_params()

    for param_name in sorted(best_params.keys()):
        print(param_name, ":", best_params[param_name])


def bagging_tuning(X, y, seed_list):
    cv = sklearn.cross_validation.StratifiedKFold(y, n_folds=5, shuffle=True, random_state=1234)

    for fold, (train_index, test_index) in enumerate(cv):
        preds = np.zeros(len(X[test_index]))
        for seed in seed_list:
            model = sklearn.linear_model.SGDClassifier(loss="log", alpha=1, l1_ratio=0.6, n_iter=100, random_state=seed)
            model.fit(X[train_index], y[train_index])
            preds += model.predict_proba(X[test_index])[:, 1]
        preds /= len(seed_list)
        print("Fold:", fold, "Score:", sklearn.metrics.log_loss(y[test_index], preds))


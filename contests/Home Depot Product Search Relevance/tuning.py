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


def rmse(y_true, y_pred):
    return np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))

def log_rmse(y_true, y_pred):
    return np.sqrt(sklearn.metrics.mean_squared_error(np.exp(y_true), np.exp(y_pred)))


def make_predictions_xgb(train, target, test):
    result = np.zeros(len(test))
    seed_list = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 123]
    for seed in seed_list:
        model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.06, max_depth=8, subsample=0.9,
                                 colsample_bytree=0.4, min_child_weight=16, seed=seed)
        model.fit(train, target)
        result += model.predict(test)

    result /= len(seed_list)

    return result

def make_predictions_xgb_log(train, target, test):
    target = np.log(target)

    result = np.zeros(len(test))
    seed_list = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 123]
    for seed in seed_list:
        model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=1.0,
                                 colsample_bytree=0.4, min_child_weight=16, seed=0)
        model.fit(train, target)
        result += model.predict(test)

    result /= len(seed_list)

    result = np.exp(result)

    return result

def ensemble_tuning(X, target):
    X = X.values
    cv = sklearn.cross_validation.StratifiedKFold(target, n_folds=5, shuffle=True, random_state=1234)

    model1 = xgb.XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=5, subsample=0.9,
                              colsample_bytree=1.0, min_child_weight=2, seed=0)

    model2 = sklearn.linear_model.ElasticNet(l1_ratio=0, alpha=1e-5, random_state=1234,
                                             max_iter=1000)

    scores = np.zeros((5, 101), dtype=np.float32)

    for fold, (train_index, test_index) in enumerate(cv):
        print("Training model1...")
        pred1 = make_predictions_xgb(X[train_index], target[train_index], X[test_index])

        print("Training model2...")
        pred2 = make_predictions_xgb_log(X[train_index], target[train_index], X[test_index])

        """
        print("Training model3...")
        bag_size = 3
        pred3 = np.zeros(pred2.shape)
        for i in range(bag_size):
            np.random.seed((i+1)*1234)
            model3 = NeuralNet(layers=layers0, input_shape=(None, num_features), dense0_num_units=100,
                               dropout_p=0.3, dense1_num_units=100, output_num_units=num_classes,
                               output_nonlinearity=softmax, update=nesterov_momentum, update_learning_rate=0.005,
                               update_momentum=0.9, eval_size=0.2, verbose=0,
                               max_epochs=300, use_label_encoder=True)

            model3.fit(X_train1[train_index], y_train[train_index])
            pred3 += model3.predict_proba(X_train1[test_index])
        pred3 /= bag_size
        """

        print("Calculating scores...")
        for alpha in np.ndindex(101):
            scores[fold][alpha] = rmse(target[test_index], 0.01*alpha[0]*pred1 + np.max(1 - 0.01*alpha[0], 0)*pred2)

        sc1 = np.mean(scores, axis=0) * 1.0 / (fold+1) * 5
        print(np.min(sc1), np.unravel_index(sc1.argmin(), sc1.shape), sc1[100], sc1[0])

    scores1 = np.mean(scores, axis=0)
    print(np.min(scores1), np.unravel_index(scores1.argmin(), scores1.shape), scores1[100], scores1[0])

    return scores

def parametr_tuning(X, y, transform_target="None"):
    if transform_target == "None":
        rmse_scorer = sklearn.metrics.make_scorer(rmse, greater_is_better=False)
    elif transform_target == "log":
        rmse_scorer = sklearn.metrics.make_scorer(log_rmse, greater_is_better=False)
        y = np.log(y)
    elif transform_target == "class":
        y = np.round(y * 100)

    param_grid = {"subsample" : [0.9]}
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.06, max_depth=8, subsample=0.9,
                             colsample_bytree=0.4, min_child_weight=16, seed=0)
    #model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=1.0,
    #                         colsample_bytree=0.4, min_child_weight=16, seed=0)
    #param_grid = {"C" : [1]}
    #model = sklearn.linear_model.LogisticRegression(penalty="l2", C=1)

    #model = sklearn.linear_model.ElasticNet(l1_ratio=0, alpha=1e-5, random_state=1234, max_iter=1000)
    #param_grid = {"alpha" : [0, 0.00001], "normalize" : [True, False]}
    #model = sklearn.neighbors.KNeighborsClassifier(n_jobs=3)
    #param_grid = {"n_neighbors" : [200, 256, 300, 350, 400, 450, 500], "metric" : ["manhattan"],
    #              "weights" : ["uniform", "distance"]}
    #model = sklearn.ensemble.ExtraTreesClassifier(n_estimators=200, max_depth=40,
    #                                              min_samples_leaf=2, n_jobs=2, random_state=1234)
    #param_grid = {"n_estimators" : [200]}

    gs = sklearn.grid_search.GridSearchCV(model, param_grid,
                                          scoring=rmse_scorer, cv=5,
                                          verbose=10, n_jobs=1)
    gs.fit(X, y)

    print("Best score is: ", gs.best_score_)
    print("Best parametrs:")

    best_params = gs.best_estimator_.get_params()

    for param_name in sorted(best_params.keys()):
        print(param_name, ":", best_params[param_name])


def align_tuning(X, y):
    model = xgb.XGBClassifier(max_depth=4, n_estimators=100, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=1, max_delta_step=3,
                              objective="multi:softprob", seed=1234)

    split = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=1, test_size=0.25,
                                                            random_state=1234)

    scores = np.zeros(201, dtype=np.float32)
    for train_index, test_index in split:
        model.fit(X[train_index], y[train_index])
        preds = model.predict_proba(X[test_index])

        argmax_vals = np.argmax(preds, axis=1)
        max_vals = np.max(preds, axis=1)

        print(sklearn.metrics.log_loss(y[test_index], preds))

        for alpha in np.ndindex(201):
            print(alpha[0])
            cur_preds = preds*(alpha[0]/100)
            cur_preds[np.arange(len(preds)), argmax_vals] = max_vals + (1 - max_vals)*(1 - alpha[0]/100)
            scores[alpha] = sklearn.metrics.log_loss(y[test_index], cur_preds)

    print(np.min(scores), np.unravel_index(scores.argmin(), scores.shape), scores[0], scores[100])

    return scores


def nulling_tuning(X, y):
    model = xgb.XGBClassifier(max_depth=4, n_estimators=100, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=1, max_delta_step=3,
                              objective="multi:softprob", seed=1234)

    split = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=1, test_size=0.25,
                                                            random_state=1234)

    scores = []
    alpha_list = np.logspace(-18, -3, 16, base=10.0)
    for train_index, test_index in split:
        model.fit(X[train_index], y[train_index])
        preds = model.predict_proba(X[test_index])

        for alpha in alpha_list:
            preds_cur = preds
            preds_cur[preds < alpha] = 0
            scores.append(sklearn.metrics.log_loss(y[test_index], preds_cur))
            print(alpha, scores[-1])

    scores = np.array(scores)

    print(np.min(scores), np.unravel_index(scores.argmin(), scores.shape), scores[0], scores[-1])

    return scores
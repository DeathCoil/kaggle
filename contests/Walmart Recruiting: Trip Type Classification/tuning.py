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
from sklearn.base import clone

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet


def ensemble_tuning(X_train1, X_train2, y_train):
    cv = sklearn.cross_validation.StratifiedKFold(y_train, n_folds=4, shuffle=True, random_state=1234)

    num_features = X_train1.shape[1]
    num_classes = len(np.unique(y_train))
    layers0 = [('input', InputLayer), ('dense0', DenseLayer), ('dropout', DropoutLayer),
               ('dense1', DenseLayer), ('output', DenseLayer)]

    model1 = xgb.XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.095,
                              subsample=0.8, colsample_bytree=1, max_delta_step=3,
                              objective="multi:softprob", seed=1234)

    model2 = sklearn.linear_model.LogisticRegression(C=0.9, random_state=1234, solver="lbfgs",
                                                     multi_class="multinomial")

    scores = np.zeros((4, 101, 101), dtype=np.float32)

    for fold, (train_index, test_index) in enumerate(cv):
        print("Training model1...")
        model1.fit(X_train1[train_index], y_train[train_index])
        pred1 = model1.predict_proba(X_train1[test_index])

        print("Training model2...")
        model2.fit(X_train2[train_index], y_train[train_index])
        pred2 = model2.predict_proba(X_train2[test_index])

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

        print("Calculating scores...")
        for alpha in np.ndindex(101, 101):
            scores[fold][alpha] = sklearn.metrics.log_loss(y_train[test_index], 0.01*alpha[0]*pred1 + 0.01*alpha[1]*pred2 + np.max(1 - 0.01*alpha[0] - 0.01*alpha[1], 0)*pred3)

        sc1 = np.mean(scores, axis = 0) * 1.0 / (fold+1) * 4
        print(np.min(sc1), np.unravel_index(sc1.argmin(), sc1.shape), sc1[100, 0], sc1[0, 100], sc1[0, 0])

    scores1 = np.mean(scores, axis = 0)
    print(np.min(scores1), np.unravel_index(scores1.argmin(), scores1.shape), scores1[100, 0], scores1[0, 100], scores1[0, 0])

    return scores

def parametr_tuning(X, y):
    #param_grid = {"subsample" : [0.8]}
    #model = xgb.XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.095,
    #                          subsample=0.8, colsample_bytree=1, max_delta_step=3,
    #                          objective="multi:softprob", seed=1234)

    #model = sklearn.linear_model.LogisticRegression(random_state=1234, solver="lbfgs", multi_class="multinomial")
    #param_grid = {"C" : [0.9], "penalty" : ["l2"]}
    #model = sklearn.neighbors.KNeighborsClassifier(n_jobs=3)
    #param_grid = {"n_neighbors" : [200, 256, 300, 350, 400, 450, 500], "metric" : ["manhattan"],
    #              "weights" : ["uniform", "distance"]}
    #model = sklearn.ensemble.ExtraTreesClassifier(n_estimators=200, max_depth=40,
    #                                              min_samples_leaf=2, n_jobs=2, random_state=1234)
    #param_grid = {"n_estimators" : [200]}

    y = y.astype(np.int32)
    #encoder = sklearn.preprocessing.LabelEncoder()
    #y = encoder.fit_transform(y).astype(np.int32)
    X = X.astype(np.float64)

    #print(X)
    #print(y)

    split = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=1,
                                                            test_size=0.25,
                                                            random_state=1234)

    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    layers0 = [('input', InputLayer), ('dense0', DenseLayer), ('dropout', DropoutLayer),
               ('dense1', DenseLayer), ('output', DenseLayer)]
    net = NeuralNet(layers=layers0, input_shape=(None, num_features), dense0_num_units=150,
                    dropout_p=0.3, dense1_num_units=150, output_num_units=num_classes,
                    output_nonlinearity=softmax, update=nesterov_momentum, update_learning_rate=0.005,
                    update_momentum=0.9, eval_size=0.01, verbose=0,
                    max_epochs=300, use_label_encoder=True)
    model = clone(net)
    param_grid = {"max_epochs" : [300]}

    """
    cv = sklearn.cross_validation.StratifiedKFold(y_train, n_folds=4,
                                                  shuffle=True,
                                                  random_state=1234)
    """

    gs = sklearn.grid_search.GridSearchCV(model, param_grid,
                                          scoring="log_loss", cv=split,
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
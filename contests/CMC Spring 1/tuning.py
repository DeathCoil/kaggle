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

from sklearn.base import clone

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet


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

def ensemble_tuning(X, target, ranking=True, load_list=[]):
    seed_list = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 123]
    cv = sklearn.cross_validation.StratifiedKFold(target, n_folds=5, shuffle=True, random_state=1234)
    scores = np.zeros((5, 101, 101), dtype=np.float32)

    for fold, (train_index, test_index) in enumerate(cv):
        print("Training model1...")
        if "linear" not in load_list:
            model1 = sklearn.linear_model.SGDClassifier(loss="log", alpha=0.01, l1_ratio=0, n_iter=100)
            model1.fit(X[train_index], target[train_index])
            pickle.dump(model1, open("level1/linear/lr_fold_"+str(fold)+".pkl", "wb"))
        else:
            model1 = pickle.load(open("level1/linear/lr_fold_"+str(fold)+".pkl", "rb"))
        pred1 = model1.predict_proba(X[test_index])[:, 1]


        print("Training model2...")
        pred2 = np.zeros(len(pred1))
        for i in range(10):
            if "xgb" not in load_list:
                model2 = xgb.XGBClassifier(n_estimators=500, max_depth=4, colsample_bytree=0.6,
                                           subsample=0.8, learning_rate=0.09, seed=seed_list[i])
                model2.fit(X[train_index], target[train_index])
                pickle.dump(model2, open("level1/xgb/xgb_fold_"+str(fold)+"_n_"+str(i)+".pkl", "wb"))
            else:
                model2 = pickle.load(open("level1/xgb/xgb_fold_"+str(fold)+"_n_"+str(i)+".pkl", "rb"))
            pred2 += model2.predict_proba(X[test_index])[:, 1]
        pred2 /= 10



        print("Training model3...")
        pred3 = np.zeros(pred2.shape)
        for i in range(10):
            if "nn" not in load_list:
                np.random.seed(seed_list[i])
                num_classes = 2
                layers0 = [('input', InputLayer), ('dense0', DenseLayer), ('dropout1', DropoutLayer),
                           ('dense1', DenseLayer), ('dropout2', DropoutLayer),
                           ('dense2', DenseLayer), ('output', DenseLayer)]
                model3 = NeuralNet(layers=layers0, input_shape=(None, (X[train_index]).shape[1]), dense0_num_units=150,
                                   dropout1_p=0.4, dense1_num_units=150,
                                   dropout2_p=0.4, dense2_num_units=150,
                                   output_num_units=num_classes,
                                   output_nonlinearity=softmax, update=nesterov_momentum, update_learning_rate=0.001,
                                   update_momentum=0.9, eval_size=0.01, verbose=0,
                                   max_epochs=100, use_label_encoder=True)

                model3.fit(X[train_index], target[train_index])
                pickle.dump(model3, open("level1/nn/nn_fold_"+str(fold)+"_n_"+str(i)+".pkl", "wb"))
            else:
                model3 = pickle.load(open("level1/nn/nn_fold_"+str(fold)+"_n_"+str(i)+".pkl", "rb"))
            pred3 += model3.predict_proba(X[test_index])[:, 1]
        pred3 /= 10

        if ranking:
            pred1 = scipy.stats.rankdata(pred1)
            pred2 = scipy.stats.rankdata(pred2)
            pred3 = scipy.stats.rankdata(pred3)

        print("Calculating scores...")
        for alpha in np.ndindex(101, 101):
            scores[fold][alpha] = sklearn.metrics.roc_auc_score(target[test_index],
                                                                0.01*alpha[0]*pred1 + 0.01*alpha[1]*pred2 + np.max(1 - 0.01*alpha[0] - 0.01*alpha[1], 0)*pred3)

        sc1 = np.mean(scores, axis=0) * 1.0 / (fold+1) * 5
        print(np.max(sc1), np.unravel_index(sc1.argmax(), sc1.shape), sc1[100, 0], sc1[0, 100], sc1[0, 0])

    scores1 = np.mean(scores, axis=0)
    print(np.max(scores1), np.unravel_index(scores1.argmax(), scores1.shape), scores1[100, 0], scores1[0, 100], scores1[0, 0])

    return scores

def parametr_tuning(X, y, param_grid):

    cv = sklearn.cross_validation.KFold(len(y), n_folds=5, random_state=1234)
    #model = xgb.XGBClassifier(n_estimators=500, max_depth=4, colsample_bytree=0.6, subsample=0.8,
    #                          learning_rate=0.09)

    model = sklearn.linear_model.SGDClassifier(loss="log", alpha=0.01, l1_ratio=0, n_iter=100)
    """
    num_classes = 2
    layers0 = [('input', InputLayer), ('dense0', DenseLayer), ('dropout1', DropoutLayer),
               ('dense1', DenseLayer), ('dropout2', DropoutLayer),
               ('dense2', DenseLayer), ('output', DenseLayer)]
    net = NeuralNet(layers=layers0, input_shape=(None, X.shape[1]), dense0_num_units=150,
                    dropout1_p=0.4, dense1_num_units=150,
                    dropout2_p=0.4, dense2_num_units=150,
                    output_num_units=num_classes,
                    output_nonlinearity=softmax, update=nesterov_momentum, update_learning_rate=0.001,
                    update_momentum=0.9, eval_size=0.01, verbose=10,
                    max_epochs=100, use_label_encoder=True)
    model = clone(net)
    """

    gs = sklearn.grid_search.GridSearchCV(model, param_grid, scoring='roc_auc', cv=cv, verbose=10,
                                          n_jobs=1)
    gs.fit(X, y)

    print("Best score is: ", gs.best_score_)
    print("Best parametrs:")

    best_params = gs.best_estimator_.get_params()

    for param_name in sorted(best_params.keys()):
        print(param_name, ":", best_params[param_name])
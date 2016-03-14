import numpy as np
import sklearn.feature_extraction.text
import sklearn.grid_search
import sklearn.linear_model
import xgboost as xgb
import pickle
import scipy

import input_output as io
import feature_extraction as fe
import tuning as tuning

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet


def make_predictions(train, target, test, ranking=False, load_list=[]):
    result = np.zeros(len(test))
    seed_list = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 123]


    print("Training model1...")
    if "linear" not in load_list:
        model1 = sklearn.linear_model.SGDClassifier(loss="log", alpha=0.01, l1_ratio=0, n_iter=100)
        model1.fit(train, target)
        pickle.dump(model1, open("final_models/linear/lr.pkl", "wb"))
    else:
        model1 = pickle.load(open("final_models/linear/lr.pkl", "rb"))
    pred1 = model1.predict_proba(test)[:, 1]


    print("Training model2...")
    pred2 = np.zeros(len(pred1))
    for i in range(10):
        if "xgb" not in load_list:
            model2 = xgb.XGBClassifier(n_estimators=500, max_depth=4, colsample_bytree=0.6,
                                       subsample=0.8, learning_rate=0.09, seed=seed_list[i])
            model2.fit(train, target)
            pickle.dump(model2, open("final_models/xgb/xgb_n_"+str(i)+".pkl", "wb"))
        else:
            model2 = pickle.load(open("final_models/xgb/xgb_n_"+str(i)+".pkl", "rb"))
        pred2 += model2.predict_proba(test)[:, 1]
    pred2 /= 10


    print("Training model3...")
    pred3 = np.zeros(len(pred2))
    for i in range(10):
        if "nn" not in load_list:
            np.random.seed(seed_list[i])
            num_classes = 2
            layers0 = [('input', InputLayer), ('dense0', DenseLayer), ('dropout1', DropoutLayer),
                       ('dense1', DenseLayer), ('dropout2', DropoutLayer),
                       ('dense2', DenseLayer), ('output', DenseLayer)]
            model3 = NeuralNet(layers=layers0, input_shape=(None, train.shape[1]), dense0_num_units=150,
                               dropout1_p=0.4, dense1_num_units=150,
                               dropout2_p=0.4, dense2_num_units=150,
                               output_num_units=num_classes,
                               output_nonlinearity=softmax, update=nesterov_momentum, update_learning_rate=0.001,
                               update_momentum=0.9, eval_size=0.01, verbose=0,
                               max_epochs=100, use_label_encoder=True)

            model3.fit(train, target)
            pickle.dump(model3, open("final_models/nn/nn_n_"+str(i)+".pkl", "wb"))
        else:
            model3 = pickle.load(open("final_models/nn/nn_n_"+str(i)+".pkl", "rb"))
        pred3 += model3.predict_proba(test)[:, 1]
    pred3 /= 10


    if ranking:
        pred1 = scipy.stats.rankdata(pred1)
        pred2 = scipy.stats.rankdata(pred2)
        pred3 = scipy.stats.rankdata(pred3)

    result = 0.21*pred1 + 0.47*pred2 + 0.32*pred3

    return result


train, test, target, test_index = io.load_data()
train, test, target = fe.preprocess_data(train, test, target)

tuning.parametr_tuning(train, target, param_grid={"alpha": [0.01]})
#tuning.ensemble_tuning(train, target, ranking=True, load_list=["linear", "xgb"])

"""
result = make_predictions(train, target, test, ranking=True, load_list=["linear", "xgb", "nn"])
io.save_result(test_index, result)
"""
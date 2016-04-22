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


def make_predictions(train, target, test, load_list=[]):
    result = np.zeros(len(test))
    seed_list = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 123]


    print("Training model1...")
    if "rf_entropy" not in load_list:
        model1 = sklearn.ensemble.RandomForestClassifier(n_estimators=2000, max_depth=8, criterion="entropy", bootstrap=False,
                                                             min_samples_leaf=4, min_samples_split=2, random_state=1234)
        model1.fit(train, target)
        pickle.dump(model1, open("final_models/rf/rf_entropy.pkl", "wb"))
    else:
        model1 = pickle.load(open("final_models/rf/rf_entropy.pkl", "rb"))
    pred1 = model1.predict_proba(test)[:, 1]


    print("Training model2...")
    pred2 = np.zeros(len(pred1))
    for i in range(10):
        if "xgb" not in load_list:
            model2 = xgb.XGBClassifier(n_estimators=100, max_depth=3, colsample_bytree=0.9, subsample=1,
                                       learning_rate=0.1, seed=seed_list[i])
            model2.fit(train, target)
            pickle.dump(model2, open("final_models/xgb/xgb_n_"+str(i)+".pkl", "wb"))
        else:
            model2 = pickle.load(open("final_models/xgb/xgb_n_"+str(i)+".pkl", "rb"))
        pred2 += model2.predict_proba(test)[:, 1]
    pred2 /= len(seed_list)

    result = 0.32*pred1 + 0.68*pred2

    return result


def main():
    train, test, target, test_index = io.load_data()
    train, test, target = fe.preprocess_data(train, test, target)

    #tuning.tune_xgboost(train, target, load_list=[])
    #tuning.parametr_tuning(train, target, param_grid={})
    #tuning.ensemble_tuning(train, target, load_list=[])

    model = sklearn.ensemble.RandomForestClassifier(n_estimators=2000, max_depth=8, criterion="entropy", bootstrap=False,
                                                    min_samples_leaf=4, min_samples_split=2, random_state=1234)

    model.fit(train, target)
    result = model.predict_proba(test)[:, 1]

    """
    result = make_predictions(train, target, test, load_list=["rf_entropy", "xgb"])
    """
    io.save_result(test_index, result)


if __name__ == "__main__":
    main()
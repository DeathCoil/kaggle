import pandas as pd
import numpy as np
import datetime as datetime
import sklearn.cross_validation


def load_data():
    train = pd.read_csv("input/train.csv")
    test = pd.read_csv("input/test.csv")

    return train, test


def save_result(index, trip_type_nums, result):
    df = pd.DataFrame({"VisitNumber": index})
    for i, trip_num in enumerate(trip_type_nums):
        df["TripType_" + str(trip_num)] = result[:, i]
    df.to_csv("output/" + str(datetime.datetime.now()), float_format="%0.7f", index=False)


def dump_train_results(X_train, target, model, name, n_folds=5):
    kf = sklearn.cross_validation.KFold(X_train.shape[0], n_folds=n_folds,
                                        shuffle=True, random_state=1234)
    preds = np.zeros(target.shape[0])

    for train_index, test_index in kf:
        model.fit(X_train[train_index], target[train_index])
        preds[test_index] = model.predict_proba(X_train[test_index])[:, 1]

    preds.dump(name)
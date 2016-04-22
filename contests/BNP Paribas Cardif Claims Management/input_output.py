import pandas as pd
import numpy as np
import datetime as datetime
import sklearn.cross_validation


def load_data(drop="drop"):
    train = pd.read_csv("input/train.csv")
    test = pd.read_csv("input/test.csv")
    target = train["target"].values.ravel()

    test_index = test["ID"].values.ravel()

    if drop == "ext":
        drop_list=['v91','v1', 'v8', 'v10', 'v15', 'v17', 'v25', 'v29', 'v34', 'v41', 'v46', 'v54', 'v64', 'v67', 'v97', 'v105', 'v111', 'v122']
        train.drop(["ID", "target"] + drop_list, axis=1, inplace=True)
        test.drop(["ID"] + drop_list, axis=1, inplace=True)
        
        return train, test, target, test_index


    train.drop(['ID','target',    "v8","v23","v25","v36","v37","v46",
                      "v51","v53","v54","v63","v73","v81",
                      "v82","v89","v92","v95","v105","v107",
                      "v108","v109","v116","v117","v118",
                      "v119","v123","v124","v128"], axis=1, inplace=True)
    test.drop(['ID',    "v8","v23","v25","v36","v37","v46",
                      "v51","v53","v54","v63","v73","v81",
                      "v82","v89","v92","v95","v105","v107",
                      "v108","v109","v116","v117","v118",
                      "v119","v123","v124","v128"], axis=1, inplace=True)

    return train, test, target, test_index


def save_result(index, result):
    df = pd.DataFrame({"ID": index, "PredictedProb": result})
    df.to_csv("output/" + str(datetime.datetime.now()), float_format="%0.7f", index=False)
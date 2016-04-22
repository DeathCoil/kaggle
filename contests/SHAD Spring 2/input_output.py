import pandas as pd
import numpy as np
import datetime as datetime


def load_data():
    train = pd.read_csv("input/X.train.csv")
    target = pd.read_csv("input/y.train.csv").values.ravel()
    test = pd.read_csv("input/X.test.csv")

    test_index = np.arange(len(test))

    return train, test, target, test_index


def save_result(index, result):
    df = pd.DataFrame({"Id": index, "Prediction": result})
    df.to_csv("output/" + str(datetime.datetime.now()), float_format="%0.7f", index=False)
import pandas as pd
import numpy as np
import datetime as datetime
import sklearn.cross_validation


def load_data():
    train = pd.read_csv("input/train.csv", encoding="ISO-8859-1")
    test = pd.read_csv("input/test.csv", encoding="ISO-8859-1")
    colors = pd.read_csv("input/colors.csv", header=None)
    colors = list(map(lambda x: x.lower(), colors.iloc[:, 1].values))

    product_descriptions = pd.read_csv("input/product_descriptions.csv", encoding="ISO-8859-1")
    train = pd.merge(train, product_descriptions, on="product_uid")
    test = pd.merge(test, product_descriptions, on="product_uid")


    attributes = pd.read_csv("input/attributes.csv", encoding="ISO-8859-1")
    attributes["name"] = attributes["name"].astype("str")
    attributes["value"] = attributes["value"].astype("str")

    target = train["relevance"].astype(np.float32)
    train.drop(["relevance"], axis=1, inplace=True)
    test_index = test["id"]

    return train, test, target, attributes, test_index


def save_result(index, result):
    result[result > 3] = 3
    result[result < 1] = 1
    df = pd.DataFrame({"id": index, "relevance": result})
    df.to_csv("output/" + str(datetime.datetime.now()), float_format="%0.7f", index=False)


def dump_train_results(X_train, target, model, name, n_folds=5):
    kf = sklearn.cross_validation.KFold(X_train.shape[0], n_folds=n_folds,
                                        shuffle=True, random_state=1234)
    preds = np.zeros(target.shape[0])

    for train_index, test_index in kf:
        model.fit(X_train[train_index], target[train_index])
        preds[test_index] = model.predict_proba(X_train[test_index])[:, 1]

    preds.dump(name)
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.ensemble


def drop_correlated(train, test, threshold=1):
    train_ = train.copy(deep=True)
    train_.fillna(2)

    columns = []
    for i in range(332, 1332):
        columns += ["V" + str(i)]

    corr_pearson = train_[columns].corr(method="pearson").values
    corr_pearson[np.diag_indices_from(corr_pearson)] = 0
    f1, f2 = np.where(np.abs(corr_pearson) > threshold)
    columns_ = np.ones(corr_pearson.shape[0])

    for x, y in zip(f1, f2):
        if columns_[x] and columns_[y]:
            columns_[x] = 0

    columns_ = list(map(lambda x: "V" + x, (np.where(1 - columns_)[0] + 332).astype("str")))

    print(columns_)

    train.drop(columns_, axis=1, inplace=True)
    test.drop(columns_, axis=1, inplace=True)
    return train, test


def drop_features(data):
    columns = data.columns[list(map(lambda x: x[0] == "V" and int(x[1:]) >= 332, data.columns))]

    data.drop(columns, axis=1, inplace=True)
    return data


def add_missing_indicator(data):
    data["missing_sum"] = np.sum(np.isnan(data), axis=1)

    return data


def add_allele_features(data):
   allele_columns = data.columns[list(map(lambda x: x[0] == "V" and int(x[1:]) >= 332, data.columns))]
   #for col_name in allele_columns:
       #data["A" + col_name[1:] + "_p(1-p)"] = data[col_name]*(1 -  data[col_name])
       #data["A" + col_name[1:] + "_1-p"] = 1 - data[col_name]

   return data


def add_features(data):
    data = add_missing_indicator(data)
    data = add_allele_features(data)
    return data


def pca_features(train, test, n_components=2):
    train_ = train.copy(deep=True)
    test_ = test.copy(deep=True)

    train_.fillna(-1, inplace=True)
    test_.fillna(-1, inplace=True)

    pca = sklearn.decomposition.PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_)
    test_pca = pca.transform(test_)
    
    return train_pca, test_pca

def input_missing(train, test):

    imp = sklearn.preprocessing.Imputer()
    train = imp.fit_transform(train)
    test = imp.transform(test)

    return train, test


def scale_data(train, test):
    scaler = sklearn.preprocessing.StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    return train, test


def shuffle_train(train, target):
    idx = np.arange(len(train))
    np.random.seed(1234)
    np.random.shuffle(idx)
    train = train.iloc[idx]
    target = target[idx]

    return train, target


def do_one_hot(train_, test_):
    data_types = pd.read_csv("input/MetaData.csv")
    cat_columns = data_types[np.logical_or((data_types["Column Type"] == "Category"),
                                           (data_types["Column Type"] == "Ordered Category"))]["varnum"].values

    train = train_.copy(deep=True)
    test = test_.copy(deep=True)

    train.fillna(12345, inplace=True)
    test.fillna(12345, inplace=True)


    le = sklearn.preprocessing.LabelEncoder()
    train_cat = le.fit_transform(train[cat_columns[0]].values)
    test_cat = le.fit_transform(test[cat_columns[0]].values)
    for i in range(1, len(cat_columns)):
        le.fit((np.concatenate((train[cat_columns[i]], test[cat_columns[i]]))))
        train_cat = np.column_stack((train_cat, le.transform(train[cat_columns[i]].values)))
        test_cat = np.column_stack((test_cat, le.transform(test[cat_columns[i]].values)))

    ohe = sklearn.preprocessing.OneHotEncoder(sparse=False)
    ohe.fit(np.concatenate((train_cat, test_cat)))
    train_cat = ohe.transform(train_cat)
    test_cat = ohe.transform(test_cat)

    return train_cat, test_cat


def take_important(train, target, test, threshold=0.001):
    model = sklearn.ensemble.RandomForestClassifier(n_estimators=2000, max_depth=8, criterion="entropy", bootstrap=False,
                                                    min_samples_leaf=4, min_samples_split=2, random_state=1234)

    model.fit(train, target)
    importance = model.feature_importances_
    print(np.sum(importance > 0.001))

    return train[:, importance > 0.001], test[:, importance > 0.001]

def preprocess_data(train, test, target):
    train, target = shuffle_train(train, target)

    train = add_features(train)
    test = add_features(test)

    train_cat, test_cat = do_one_hot(train, test)
    train_pca, test_pca = pca_features(train, test, n_components=16)

    train = drop_features(train)
    test = drop_features(test)

    train, test = input_missing(train, test)
    train, test = scale_data(train, test)

    train = np.column_stack((train, train_cat))
    test = np.column_stack((test, test_cat))

    train = np.column_stack((train, train_pca))
    test = np.column_stack((test, test_pca))

    train, test = take_important(train, target, test)

    return train, test, target

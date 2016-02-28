import numpy as np
import pandas as pd
import collections
import sklearn.cross_validation
import sklearn.preprocessing


def extract_counts(data):
    visits = data["VisitNumber"].values
    unique_visits = np.unique(visits)
    vals = data["ScanCount"].values
    counts = []

    for visit in unique_visits:
        counts.append(vals[visits == visit])

    return np.array(counts)


def concat(data, name, counts):
    visits = data["VisitNumber"].values
    unique_visits = np.unique(visits)
    vals = data[name].values
    cur_list = []

    for i, visit in enumerate(unique_visits):
        cur_vals = map(lambda x: str(x).replace(".0", ""), vals[visits == visit])
        counter = collections.Counter()
        for val, cnt in zip(cur_vals, counts[i]):
            counter[val] += int(cnt)

        cur_list.append(" ".join(counter.elements()))

    return cur_list


def concat_lines(data):
    counts = extract_counts(data)
    np.random.seed(1234)
    upc = concat(data, "Upc", counts)
    department = concat(data, "DepartmentDescription", counts)
    data["DepartmentDescription"] = data["DepartmentDescription"].apply(lambda x: str(x).replace(" ", "_"))
    department_ = concat(data, "DepartmentDescription", counts)
    fineline = concat(data, "FinelineNumber", counts)

    data.drop_duplicates(subset=["VisitNumber"],inplace=True)
    data.drop(["Upc", "ScanCount", "DepartmentDescription",
                "FinelineNumber"], axis=1, inplace=True)

    data["Upc"] = upc
    data["DepartmentDescription"] = department
    data["DepartmentDescription_"] = department_
    data["FinelineNumber"] = fineline
    data["Counts"] = counts

    data.reset_index(drop=True, inplace=True)

    return data


def add_features(data):
    data["Num_items"] = data["Counts"].apply(lambda x: np.sum(np.abs(x)))
    data["Num_pos_items"] = data["Counts"].apply(lambda x: np.sum(x[x > 0]))
    data["Num_neg_items"] = data["Counts"].apply(lambda x: np.sum(x[x < 0]))
    data["Max_count"] = data["Counts"].apply(lambda x: np.max(x))
    data["Num_pos-neg_items"] = data["Counts"].apply(lambda x: np.sum(x))
    data["UniqueDepartmentsCounter_"] = data["DepartmentDescription_"].apply(lambda x: len(np.unique(x.split())))
    data["UniqueDepartmentsCounter"] = data["DepartmentDescription"].apply(lambda x: len(np.unique(x.split())))
    data["UniqueFinelineNumberCounter"] = data["FinelineNumber"].apply(lambda x: len(np.unique(x.split())))
    data["UniqueUpcCounter"] = data["Upc"].apply(lambda x: len(np.unique(x.split())))
    data["ratio"] = data["DepartmentDescription"].apply(lambda x: x.count("DepartmentNan"))/data["Num_items"]



    return data


def extract_features(data):
    data = add_features(data)

    columns = ["Friday", "Monday", "Saturday", "Sunday", "Thursday",
               "Tuesday", "Wednesday", "Num_pos_items", "Num_neg_items",
               "Num_items", "Num_pos-neg_items", "Max_count", "UniqueDepartmentsCounter_",
               "UniqueDepartmentsCounter", "UniqueFinelineNumberCounter", "UniqueUpcCounter",
               "VisitNumber", "ratio"]

    return data[columns]


def preprocess_data(train_, test_, load=True):
    if load:
        train = pd.read_pickle("cache/df_train")
        test = pd.read_pickle("cache/df_test")
        target = np.load("cache/target")
        test_index = np.load("cache/test_index")
    else:
        train = train_.copy(deep=True)
        test = test_.copy(deep=True)

        train["Upc"][train["Upc"].isnull()] = "UpcNan"
        test["Upc"][test["Upc"].isnull()] = "UpcNan"

        train["FinelineNumber"][train["FinelineNumber"].isnull()] = "FinelineNan"
        test["FinelineNumber"][test["FinelineNumber"].isnull()] = "FinelineNan"

        train["DepartmentDescription"][train["DepartmentDescription"].isnull()] = "DepartmentNan"
        test["DepartmentDescription"][test["DepartmentDescription"].isnull()] = "DepartmentNan"


        train = concat_lines(train)
        test = concat_lines(test)

        train = add_features(train)
        test = add_features(test)

        lb = sklearn.preprocessing.LabelBinarizer()

        days = np.unique(train["Weekday"])

        binarized = lb.fit_transform(train["Weekday"])
        for i, day in enumerate(days):
            train[day] = binarized[:, i]

        binarized = lb.transform(test["Weekday"])
        for i, day in enumerate(days):
            test[day] = binarized[:, i]

        target = train["TripType"].values
        test_index = test["VisitNumber"].values

        train.drop(["TripType", "Weekday"], axis=1, inplace=True)
        test.drop(["Weekday"], axis=1, inplace=True)


        train.to_pickle("cache/df_train")
        test.to_pickle("cache/df_test")

        target.dump("cache/target")
        test_index.dump("cache/test_index")

    return train, target, test, test_index

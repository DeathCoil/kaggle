import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.calibration
import scipy.sparse

import sklearn.linear_model
import sklearn.metrics

import input_output as io


class addNearestNeighbourLinearFeatures:
    def __init__(self, n_neighbours=1, max_elts=None, verbose=True, random_state=None):
        self.rnd=random_state
        self.n=n_neighbours
        self.max_elts=max_elts
        self.verbose=verbose
        self.neighbours=[]
        self.clfs=[]
        
    def fit(self,train,y):
        if self.rnd!=None:
            np.random.seed(self.rnd)
        if self.max_elts==None:
            self.max_elts=len(train.columns)
        list_vars=list(train.columns)
        np.random.shuffle(list_vars)
        
        lastscores=np.zeros(self.n)+1e15

        for elt in list_vars[:self.n]:
            self.neighbours.append([elt])
        list_vars=list_vars[self.n:]
        
        for elt in list_vars:
            indice=0
            scores=[]
            for elt2 in self.neighbours:
                if len(elt2)<self.max_elts:
                    clf=sklearn.linear_model.LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1) 
                    clf.fit(train[elt2+[elt]], y)
                    scores.append(sklearn.metrics.log_loss(y,clf.predict(train[elt2 + [elt]])))
                    indice=indice+1
                else:
                    scores.append(lastscores[indice])
                    indice=indice+1
            gains=lastscores-scores
            if gains.max()>0:
                temp=gains.argmax()
                lastscores[temp]=scores[temp]
                self.neighbours[temp].append(elt)

        indice=0
        for elt in self.neighbours:
            clf=sklearn.linear_model.LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1) 
            clf.fit(train[elt], y)
            self.clfs.append(clf)
            if self.verbose:
                print(indice, lastscores[indice], elt)
            indice=indice+1
                    
    def transform(self, train):
        indice=0
        for elt in self.neighbours:
            train['_'.join(pd.Series(elt).sort_values().values)]=self.clfs[indice].predict(train[elt])
            indice=indice+1
        return train
    
    def fit_transform(self, train, y):
        self.fit(train, y)
        return self.transform(train)


cat_columns = []

num_columns = ['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v9', 'v10', 'v11',
               'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20',
               'v21', 'v26', 'v27', 'v28', 'v29', 'v32', 'v33', 'v34', 'v35', 'v38',
               'v39', 'v40', 'v41', 'v42', 'v43', 'v44', 'v45', 'v48', 'v49', 'v50',
               'v55', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v64', 'v65', 'v67',
               'v68', 'v69', 'v70', 'v72', 'v76', 'v77', 'v78', 'v80', 'v83', 'v84',
               'v85', 'v86', 'v87', 'v88', 'v90', 'v93', 'v94', 'v96', 'v97', 'v98',
               'v99', 'v100', 'v101', 'v102', 'v103', 'v104', 'v106', 'v111', 'v114',
               'v115', 'v120', 'v121', 'v122', 'v126', 'v127', 'v129', 'v130', 'v131']



def drop_features(data):
    columns = []

    data.drop(columns, axis=1, inplace=True)
    return data


def add_missing_indicator(data):
    data["missing_sum"] = np.sum(data.isnull(), axis=1)

    return data


def add_features(data):
    data = add_missing_indicator(data)
    columns = []
    for col in num_columns:
        if col in data.columns:
            columns += [col]
    data["0_counts"] = np.sum(data.fillna(1)[columns] < 1e-5, axis=1)
    return data


def input_missing(train, test, input_type="-999"):

    if input_type == "mean":
        imp = sklearn.preprocessing.Imputer()
        train = imp.fit_transform(train)
        test = imp.transform(test)
    elif input_type == "-999":
        train.fillna(-999, inplace=True)
        test.fillna(-999, inplace=True)
    elif input_type == "-1":
        train.fillna(-1, inplace=True)
        test.fillna(-1, inplace=True)        

    return train, test


def scale_data(train, test):
    scaler = sklearn.preprocessing.StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    return train, test


def do_one_hot(train, test, drop_cat=False):

    ohe_columns = []
    for col in cat_columns:
        if len(np.unique(train[col])) < 50:
            ohe_columns += [col]

    ohe = sklearn.preprocessing.OneHotEncoder(sparse=False)
    ohe.fit(np.concatenate((train[ohe_columns].values, test[ohe_columns].values)))
    train_cat = scipy.sparse.csc_matrix(ohe.transform(train[ohe_columns]))
    test_cat = scipy.sparse.csc_matrix(ohe.transform(test[ohe_columns]))

    if drop_cat:
        train.drop(cat_columns, axis=1, inplace=True)
        test.drop(cat_columns, axis=1, inplace=True)

    return train_cat, test_cat


def label_cats(train_, test_, nan_value=-999):
    train = train_.copy(deep=True)
    test = test_.copy(deep=True)
    train.fillna("nan", inplace=True)
    test.fillna("nan", inplace=True)

    global cat_columns

    for feature in train.columns:
        if train_.dtypes[feature] == "O":
            cat_columns += [feature]
            le = sklearn.preprocessing.LabelEncoder()
            le.fit(np.concatenate((train[feature].values.ravel(), test[feature].values.ravel())))
            train_[feature] = le.transform(train[feature])
            train_[feature][train[feature] == "nan"] = nan_value
            test_[feature] = le.transform(test[feature])
            test_[feature][test[feature] == "nan"] = nan_value

    return train_, test_


def Binarize(columnName, df, features=None):
    df[columnName] = df[columnName].astype(str)
    if(features is None):
        features = np.unique(df[columnName].values)
    for x in features:
        df[columnName+'_' + x] = df[columnName].map(lambda y:
                                                    1 if y == x else 0)
    df.drop(columnName, inplace=True, axis=1)
    return df, features


def MungeData(train, target, test):

    todrop = ['v22']

    train.drop(todrop,
               axis=1, inplace=True)
    test.drop(todrop,
              axis=1, inplace=True)

    features = train.columns
    for col in features:
        if((train[col].dtype == 'object')):
            print(col)
            train, binfeatures = Binarize(col, train)
            test, _ = Binarize(col, test, binfeatures)
            nb = sklearn.naive_bayes.BernoulliNB()
            nb.fit(train[col+'_'+binfeatures].values, target)
            train[col + "_nb"] = \
                nb.predict_proba(train[col+'_'+binfeatures].values)[:, 1]
            test[col + "_nb"] = \
                nb.predict_proba(test[col+'_'+binfeatures].values)[:, 1]
            train.drop(col+'_'+binfeatures, inplace=True, axis=1)
            test.drop(col+'_'+binfeatures, inplace=True, axis=1)

    train = train.astype(float)
    test = test.astype(float)
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)
    return train, test


def preprocess_data(train, test, target, preprocess_type=1):
    if preprocess_type == 1:
        train = add_features(train)
        test = add_features(test)

        train, test = label_cats(train, test, nan_value=-999)

        train = add_features(train)
        test = add_features(test)
        train = drop_features(train)
        test = drop_features(test)

        train, test = input_missing(train, test, input_type="-999")

    elif preprocess_type == 2:
        train, test = label_cats(train, test, nan_value=10000)
        train_cat, test_cat = do_one_hot(train, test, drop_cat=True)

        train = add_features(train)
        test = add_features(test)
        train = drop_features(train)
        test = drop_features(test)

        train, test = input_missing(train, test, input_type="mean")

        train = scipy.sparse.hstack((train, train_cat), format="csr")
        test = scipy.sparse.hstack((test, test_cat), format="csr")
    
    elif preprocess_type == 3:
        train = add_features(train)
        test = add_features(test)
        train, test = label_cats(train, test, nan_value=-999)

        train, test = MungeData(train, target, test)

        train = drop_features(train)
        test = drop_features(test)

        train, test = input_missing(train, test, input_type="-999")      
        
    elif preprocess_type == 4:
        train = add_features(train)
        test = add_features(test)

        train, test = MungeData(train, target, test)

        train = drop_features(train)
        test = drop_features(test)

        train, test = input_missing(train, test, input_type="-1") 

        scaler = sklearn.preprocessing.StandardScaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)
    
    elif preprocess_type == 5:

        train, test, target, test_index = io.load_data(drop="ext")        
   
        train, test = label_cats(train, test, nan_value=-999)

        train['v22-1']=train['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[0]))
        test['v22-1']=test['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[0]))
        train['v22-2']=train['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[1]))
        test['v22-2']=test['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[1]))
        train['v22-3']=train['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[2]))
        test['v22-3']=test['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[2]))
        train['v22-4']=train['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[3]))
        test['v22-4']=test['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[3]))

        train = add_features(train)
        test = add_features(test)
        train = drop_features(train)
        test = drop_features(test)

        train, test = input_missing(train, test, input_type="-999")        

        rnd = 12
        n_ft = 20
        max_elts = 3
        

        a=addNearestNeighbourLinearFeatures(n_neighbours=n_ft, max_elts=max_elts, verbose=True, random_state=rnd)
        a.fit(train, target)

        train = a.transform(train)
        test = a.transform(test)

    elif preprocess_type == 6:
        train = add_features(train)
        test = add_features(test)

        train, test = label_cats(train, test, nan_value=-999)

        train = add_features(train)
        test = add_features(test)
        train = drop_features(train)
        test = drop_features(test)

        train.drop(["v50"], axis=1, inplace=True)
        test.drop(["v50"], axis=1, inplace=True)
        
        train, test = input_missing(train, test, input_type="-999")

    if type(train) is pd.DataFrame:
        train = train.values
        test = test.values

    return train, test, target

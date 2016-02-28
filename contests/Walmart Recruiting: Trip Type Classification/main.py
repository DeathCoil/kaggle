import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import sklearn.feature_extraction.text
import sklearn.grid_search
import sklearn.linear_model
import sklearn.neighbors
import sklearn.ensemble
import xgboost as xgb
import sklearn.decomposition

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import input_output as io
import feature_extraction as fe
import metafeatures as mf
import tuning as tuning
import word2vec_model as w2v


def watch_errors(X_train, y_train):
    preds = np.zeros(y_train.shape[0], dtype=np.float32)
    model = sklearn.ensemble.RandomForestClassifier(max_depth=12, n_estimators=500,
                                                    min_samples_leaf=8, min_samples_split=2)

    kf = sklearn.cross_validation.KFold(X_train.shape[0], n_folds=5,
                                        shuffle=True, random_state=1234)

    for train_index, test_index in kf:
        model.fit(X_train[train_index], y_train[train_index])
        preds[test_index] = model.predict_proba(X_train[test_index])[:, 1]

    np.random.seed(1234)
    error_index = np.random.choice(np.arange(X_train.shape[0])[preds != y_train], 25)
    print(error_index)

    return preds


def make_predictions(model, X_train, target, X_test):
    model.fit(X_train, target)
    return model.predict_proba(X_test)[:, 1]


def plot_data(X_train, target, X_test):
    for feature in X_train.columns:
        plt.figure()
        plt.hist(X_train[feature][target == 1].values, 100, alpha=0.5, normed=1, facecolor='g', label="1")
        plt.hist(X_train[feature][target == 0].values, 100, alpha=0.5, normed=1, facecolor='b', label="0")
        plt.legend(loc="upper right")
        plt.title(feature)
        plt.grid(True)
        plt.show()


def plot_hist(target, preds):
    plt.figure()
    plt.hist(target, bins=len(np.unique(target)), alpha=0.5, facecolor="g", label="target")
    plt.hist(preds, bins=len(np.unique(target)), alpha=0.1, facecolor="r", label="preds")
    plt.legend()
    plt.show()

def ensemble(target):
    preds1 = np.load("ensemble_results/rf_stacked")
    preds2 = np.load("ensemble_results/linear_model")
    scores = np.zeros(101, dtype=np.float32)
    for alpha in range(0, 101):
        scores[alpha] = sklearn.metrics.roc_auc_score(target, 0.01*alpha*preds1 + np.max(1 - 0.01*alpha, 0)*preds2)
    print(np.max(scores), np.unravel_index(scores.argmax(), scores.shape), scores[0], scores[100])


def get_tfidf(train, test):
    text_train = train["FinelineNumber"].values
    text_test = test["FinelineNumber"].values

    tfidf1 = sklearn.feature_extraction.text.TfidfVectorizer(analyzer="word", ngram_range=(1, 1),
                                                             min_df=10)
    text_train_tfidf = tfidf1.fit_transform(text_train)
    text_test_tfidf = tfidf1.transform(text_test)

    tfidf4 = sklearn.feature_extraction.text.TfidfVectorizer(analyzer="word", ngram_range=(1, 1),
                                                             min_df=10, norm="l1")
    text_train_tfidf = scipy.sparse.hstack((text_train_tfidf, tfidf4.fit_transform(text_train)), format="csr")
    text_test_tfidf = scipy.sparse.hstack((text_test_tfidf, tfidf4.transform(text_test)), format="csr")

    text_train = train["DepartmentDescription"].values
    text_test = test["DepartmentDescription"].values

    tfidf2 = sklearn.feature_extraction.text.TfidfVectorizer(analyzer="word", ngram_range=(1, 1),
                                                             min_df=10)
    text_train_tfidf = scipy.sparse.hstack((text_train_tfidf, tfidf2.fit_transform(text_train)), format="csr")
    text_test_tfidf = scipy.sparse.hstack((text_test_tfidf, tfidf2.transform(text_test)), format="csr")

    tfidf3 = sklearn.feature_extraction.text.TfidfVectorizer(analyzer="char", ngram_range=(1, 3),
                                                             min_df=20)
    text_train_tfidf = scipy.sparse.hstack((text_train_tfidf, tfidf3.fit_transform(text_train)), format="csr")
    text_test_tfidf = scipy.sparse.hstack((text_test_tfidf, tfidf3.transform(text_test)), format="csr")

    return text_train_tfidf, text_test_tfidf


train, test = io.load_data()
train, target, test, test_index = fe.preprocess_data(train, test, load=True)

w2v_train, w2v_test = w2v.word2vec_features(train, test, load=True)
text_train_tfidf, text_test_tfidf = get_tfidf(train, test)
#tuning.parametr_tuning(text_train_tfidf, target)

text_train = scipy.sparse.hstack((text_train_tfidf, w2v_train), format="csr")
text_test = scipy.sparse.hstack((text_test_tfidf, w2v_test), format="csr")
linear_train, linear_test = mf.linear_model_as_feature(text_train, target, text_test, load=True)

#tuning.parametr_tuning(text_train_tfidf, target)

X_train = fe.extract_features(train).values.astype(np.float64)
X_test = fe.extract_features(test).values.astype(np.float64)

scaler = sklearn.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train1 = np.column_stack((X_train, linear_train))
X_test1 = np.column_stack((X_test, linear_test))

#tuning.parametr_tuning(X_train, target)

X_train2 = scipy.sparse.hstack((text_train, X_train), format="csr")
X_test2 = scipy.sparse.hstack((text_test, X_test), format="csr")

#tuning.parametr_tuning(X_train, target)
#scores = tuning.nulling_tuning(X_train, target)
#scores = tuning.ensemble_tuning(X_train1, X_train2, target)


print("Training xgboost model...")
preds1 = np.zeros((X_test1.shape[0], len(np.unique(target))))
seed_list = [1234, 2345, 3456, 4567, 5678]
xgb_list = []

for seed_ in seed_list:
    print(seed_)
    xgb_list.append(xgb.XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.095,
                                      subsample=0.8, colsample_bytree=1, max_delta_step=3,
                                      objective="multi:softprob", seed=seed_))
    xgb_list[-1].fit(X_train1, target)
    temp_preds = xgb_list[-1].predict_proba(X_test1)
    print(temp_preds[0])
    preds1 += temp_preds


preds1 /= len(seed_list)
io.save_result(test_index, np.unique(target), preds1)

print("Training linear model...")
model = sklearn.linear_model.LogisticRegression(C=0.9, random_state=1234, solver="lbfgs",
                                                multi_class="multinomial")
model.fit(X_train2, target)
preds2 = model.predict_proba(X_test2)
io.save_result(test_index, np.unique(target), preds2)

print("Training neural network...")
bag_size = 5
preds3 = np.zeros(preds2.shape)
for i in range(bag_size):
    np.random.seed((i+1)*1234)
    num_features = X_train1.shape[1]
    num_classes = len(np.unique(target))
    layers0 = [('input', InputLayer), ('dense0', DenseLayer), ('dropout', DropoutLayer),
               ('dense1', DenseLayer), ('output', DenseLayer)]
    model3 = NeuralNet(layers=layers0, input_shape=(None, num_features), dense0_num_units=100,
                       dropout_p=0.3, dense1_num_units=100, output_num_units=num_classes,
                       output_nonlinearity=softmax, update=nesterov_momentum, update_learning_rate=0.005,
                       update_momentum=0.9, eval_size=0.01, verbose=0,
                       max_epochs=300, use_label_encoder=True)

    model3.fit(X_train1, target)
    preds3 += model3.predict_proba(X_test1)
preds3 /= bag_size
io.save_result(test_index, np.unique(target), preds3)

preds = 0.8*preds1 + 0.1*preds2 + 0.1*preds3
io.save_result(test_index, np.unique(target), preds)
import numpy as np
import sklearn.feature_extraction.text
import scipy.io

def make_tfidf_model(train, test):
    tfidf_model = sklearn.feature_extraction.text.TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                                                  min_df=25)

    texts_train = train["product_title"].values + "." + train["product_description"].values + "." + train["search_term"].values + "."
    texts_test = test["product_title"].values + "." + test["product_description"].values + "." + test["search_term"].values + "."
    texts = np.concatenate((texts_train, texts_test))
    tfidf_model.fit(texts)

    train_tfidf_title = tfidf_model.transform(train["product_title"].values)
    scipy.io.mmwrite("tfidf/train_tfidf_title", train_tfidf_title)
    del train_tfidf_title

    train_tfidf_descr = tfidf_model.transform(train["product_description"].values)
    scipy.io.mmwrite("tfidf/train_tfidf_descr", train_tfidf_descr)
    del train_tfidf_descr

    train_tfidf_title_descr = tfidf_model.transform(train["product_title"].values + train["product_description"].values)
    scipy.io.mmwrite("tfidf/train_tfidf_title_descr", train_tfidf_title_descr)
    del train_tfidf_title_descr

    train_tfidf_attr = tfidf_model.transform(train["attr"].values)
    scipy.io.mmwrite("tfidf/train_tfidf_attr", train_tfidf_attr)
    del train_tfidf_attr

    train_tfidf_query = tfidf_model.transform(train["search_term"].values)
    scipy.io.mmwrite("tfidf/train_tfidf_query", train_tfidf_query)
    del train_tfidf_query

    test_tfidf_title = tfidf_model.transform(test["product_title"].values)
    scipy.io.mmwrite("tfidf/test_tfidf_title", test_tfidf_title)
    del test_tfidf_title

    test_tfidf_descr = tfidf_model.transform(test["product_description"].values)
    scipy.io.mmwrite("tfidf/test_tfidf_descr", test_tfidf_descr)
    del test_tfidf_descr

    test_tfidf_title_descr = tfidf_model.transform(test["product_title"].values + test["product_description"].values)
    scipy.io.mmwrite("tfidf/test_tfidf_title_descr", test_tfidf_title_descr)
    del test_tfidf_title_descr

    test_tfidf_attr = tfidf_model.transform(test["attr"].values)
    scipy.io.mmwrite("tfidf/test_tfidf_attr", test_tfidf_attr)
    del test_tfidf_attr

    test_tfidf_query = tfidf_model.transform(test["search_term"].values)
    scipy.io.mmwrite("tfidf/test_tfidf_query", test_tfidf_query)
    del test_tfidf_query


def load(name):
    if name == "query":
        train_tfidf_query = scipy.sparse.csr_matrix(scipy.io.mmread("tfidf/train_tfidf_query"))
        test_tfidf_query = scipy.sparse.csr_matrix(scipy.io.mmread("tfidf/test_tfidf_query"))
        return train_tfidf_query, test_tfidf_query

    elif name == "title":
        train_tfidf_title = scipy.sparse.csr_matrix(scipy.io.mmread("tfidf/train_tfidf_title"))
        test_tfidf_title = scipy.sparse.csr_matrix(scipy.io.mmread("tfidf/test_tfidf_title"))
        return train_tfidf_title, test_tfidf_title

    elif name == "descr":
        train_tfidf_title = scipy.sparse.csr_matrix(scipy.io.mmread("tfidf/train_tfidf_descr"))
        test_tfidf_title = scipy.sparse.csr_matrix(scipy.io.mmread("tfidf/test_tfidf_descr"))
        return train_tfidf_title, test_tfidf_title

    elif name == "title_descr":
        train_tfidf_title_descr = scipy.sparse.csr_matrix(scipy.io.mmread("tfidf/train_tfidf_title_descr"))
        test_tfidf_title_descr = scipy.sparse.csr_matrix(scipy.io.mmread("tfidf/test_tfidf_title_descr"))
        return train_tfidf_title_descr, test_tfidf_title_descr

    elif name == "attr":
        train_tfidf_attr = scipy.sparse.csr_matrix(scipy.io.mmread("tfidf/train_tfidf_attr"))
        test_tfidf_attr = scipy.sparse.csr_matrix(scipy.io.mmread("tfidf/test_tfidf_attr"))
        return train_tfidf_attr, test_tfidf_attr
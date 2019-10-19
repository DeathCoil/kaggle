import numpy as np
import gensim.models.word2vec
import nltk.tokenize
import logging


def make_word2vec_model(train, test, size=300):
    texts_train = train["product_title"].values + "." + train["product_description"].values + "." + train["search_term"].values + "."
    texts_test = test["product_title"].values + "." + test["product_description"].values + "." + test["search_term"].values + "."
    texts = np.concatenate((texts_train, texts_test))

    sentences = []
    for text in texts:
        sentences += nltk.tokenize.sent_tokenize(text)

    sentences = list(map(lambda x: nltk.tokenize.word_tokenize(x), sentences))

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)


    model = gensim.models.word2vec.Word2Vec(sentences, workers=12, size=size,
                                            min_count=10, window=10,
                                            sample=0.0001, seed=1234)
    model.init_sims(replace=True)
    model.save("w2v/basic_word2vec_model")

    return model


def texts_to_words(data, load_model=True, size=300):
    model = None
    if load_model:
        model = gensim.models.word2vec.Word2Vec.load("w2v/basic_word2vec_model")
    else:
        raise("Error")

    w2v_words = set(model.index2word)
    words = list(map(lambda x: nltk.tokenize.word_tokenize(x), data))

    feature = np.zeros((len(words), size))

    for i, words in enumerate(words):
        n = 0
        for word in words:
            if word in w2v_words:
                feature[i] += model[word]
                n += 1
        if n:
            feature[i] /= n

    return feature


def word2vec_load_features():
    train_title = np.load("w2v/word2vec_train_title")
    train_descr = np.load("w2v/word2vec_train_descr")
    train_query = np.load("w2v/word2vec_train_query")

    test_title = np.load("w2v/word2vec_test_title")
    test_descr = np.load("w2v/word2vec_test_descr")
    test_query = np.load("w2v/word2vec_test_query")

    return train_title, train_descr, train_query, test_title, test_descr, test_query


def word2vec_features(train, test, size=300, load=True):
    if load:
        return word2vec_load_features()

    train_title = texts_to_words(train["product_title"], size=size)
    train_descr = texts_to_words(train["product_description"], size=size)
    train_query = texts_to_words(train["search_term"], size=size)

    test_title = texts_to_words(test["product_title"], size=size)
    test_descr = texts_to_words(test["product_description"], size=size)
    test_query = texts_to_words(test["search_term"], size=size)

    train_title.dump("w2v/word2vec_train_title")
    train_descr.dump("w2v/word2vec_train_descr")
    train_query.dump("w2v/word2vec_train_query")

    test_title.dump("w2v/word2vec_test_title")
    test_descr.dump("w2v/word2vec_test_descr")
    test_query.dump("w2v/word2vec_test_query")

    return train_title, train_descr, train_query, test_title, test_descr, test_query
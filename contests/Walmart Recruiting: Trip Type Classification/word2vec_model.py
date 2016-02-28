import numpy as np
import gensim.models.word2vec
import logging

def make_word2vec_model(train_, test_, size=100):
    train = train_.copy()
    test = test_.copy()
    texts = np.concatenate((train.values, test.values))

    texts = list(map(lambda x: x.split(), texts))

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    model = gensim.models.word2vec.Word2Vec(texts, workers=12, size=size,
                                            min_count=10, window=100,
                                            sample=0.0001, seed=1234)
    model.init_sims(replace=True)
    model.save("w2v/basic_word2vec_model")

    return model


def texts_to_words(data, size=100, load_model=True):
    model = None
    if load_model:
        model = gensim.models.word2vec.Word2Vec.load("w2v/basic_word2vec_model")
    else:
        raise("Error")

    w2v_words = set(model.index2word)
    sentences = list(map(lambda x: x.split(), data))

    feature = np.zeros((len(sentences), size))

    for i, words in enumerate(sentences):
        n = 0
        for word in words:
            if word in w2v_words:
                feature[i] += model[word]
                n += 1
        if n:
            feature[i] /= n

    return feature


def word2vec_features(train, test, load=True):
    if load:
        feature_train = np.load("w2v/feature_train")
        feature_test = np.load("w2v/feature_test")
        return feature_train, feature_test
    else:
        make_word2vec_model(train["FinelineNumber"], test["FinelineNumber"], size=200)
        feature_train = texts_to_words(train["FinelineNumber"], size=200, load_model=True)
        feature_test = texts_to_words(test["FinelineNumber"], size=200, load_model=True)

        make_word2vec_model(train["DepartmentDescription"], test["DepartmentDescription"], size=150)
        feature_train = np.column_stack((feature_train, texts_to_words(train["DepartmentDescription"], size=150, load_model=True)))
        feature_test = np.column_stack((feature_test, texts_to_words(test["DepartmentDescription"], size=150, load_model=True)))

        make_word2vec_model(train["Upc"], test["Upc"], size=75)
        feature_train = np.column_stack((feature_train, texts_to_words(train["Upc"], size=75, load_model=True)))
        feature_test = np.column_stack((feature_test, texts_to_words(test["Upc"], size=75, load_model=True)))

        feature_train.dump("w2v/feature_train")
        feature_test.dump("w2v/feature_test")
        return feature_train, feature_test
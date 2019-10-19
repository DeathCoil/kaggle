import numpy as np
import pandas as pd
import re as re
import nltk
import sklearn
import scipy

from itertools import chain, combinations

import word2vec_model as w2v
import tfidf as tfidf


stemmer = nltk.stem.porter.PorterStemmer()


def extract_color(text, colors):
    extracted = []
    for word in nltk.tokenize.word_tokenize(text):
        for color in colors:
            if word == color:
                extracted.append(word)

    return " ".join(extracted)



def last_word_in(query_, text):
    word = nltk.tokenize.word_tokenize(text)[-1]
    query = " " + re.sub(r"[^a-z\-\&]", " ", query_) + " "
    if query.find(" " + word + " ") >= 0:
        return 1
    else:
        return 0


def str_whole_word(str1, str2, i_):
    str1, str2 = str1.lower().strip(), str2.lower().strip()
    cnt = 0

    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt


def powerset(lst, minn, maxn):
    return chain.from_iterable(combinations(lst,n) for n in range(minn, maxn+1))


def add_ngram_query_exact_match(query_, text_, minn, maxn):
    text = " " + re.sub(r"[^a-z0-9\-\&]", r" ", text_.lower()) + " "
    query = " " + re.sub(r"[^a-z0-9\-\&]", r" ", query_.lower()) + " "
    querys = powerset(nltk.tokenize.word_tokenize(query), minn, maxn)
    counter = 0
    length = 0
    for q in querys:
        if text.find(" " + " ".join(q) + " ") >= 0:
            counter += 1
        length += 1
    return (counter + 1)/(length + 1)


def add_color_features(data):

    colors = np.array(list(np.load("cache/color_set").tolist()))

    data["query_colors"] = data["search_term"].apply(lambda x: extract_color(x, colors)).astype("str")
    data["title_colors"] = data["product_title"].apply(lambda x: extract_color(x, colors)).astype("str")
    data["description_colors"] = data["product_description"].apply(lambda x: extract_color(x, colors)).astype("str")

    data["query_product_colors"] = (data["query_colors"] + "\t" + data["title_colors"] +\
                                   "\t"+data["description_colors"] + "\t" + data["colors"]).astype("str")

    data["colors_in_title"] = data["query_product_colors"].apply(lambda x: count_text1_in_text2(x.split("\t")[0],
                                                                                              x.split("\t")[1]))
    data["colors_in_description"] = data["query_product_colors"].apply(lambda x: count_text1_in_text2(x.split("\t")[0],
                                                                                                    x.split("\t")[2]))
    data["colors_in_colors"] = data["query_product_colors"].apply(lambda x: count_text1_in_text2(x.split("\t")[0],
                                                                                               x.split("\t")[3]))
    data["colors_in_product_info"] = data["colors_in_title"] + data["colors_in_description"]

    data["query_color_counter"] = data["query_colors"].apply(lambda x: len(nltk.tokenize.word_tokenize(x)))
    data["title_color_counter"] = data["title_colors"].apply(lambda x: len(nltk.tokenize.word_tokenize(x)))
    data["description_color_counter"] = data["description_colors"].apply(lambda x: len(nltk.tokenize.word_tokenize(x)))
    data["info_color_counter"] = data["title_color_counter"] + data["description_color_counter"]
    data["color_color_counter"] = data["colors"].apply(lambda x: len(nltk.tokenize.word_tokenize(x)))

    return data

def add_features(data):
    #data = add_color_features(data)
    return data


def add_w2v_sim_features(train, test):
    train_title, train_descr, train_query, test_title, test_descr, test_query = w2v.word2vec_load_features()

    train["w2v_sim_l1_query_title"] = np.linalg.norm(train_query - train_title, ord=1, axis=1)
    train["w2v_sim_l1_query_descr"] = np.linalg.norm(train_query - train_descr, ord=1, axis=1)
    train["w2v_sim_l2_query_title"] = np.linalg.norm(train_query - train_title, ord=2, axis=1)
    train["w2v_sim_l2_query_descr"] = np.linalg.norm(train_query - train_descr, ord=2, axis=1)
    train["w2v_sim_cos_query_title"] = np.sum(train_query*train_title, axis=1)/np.linalg.norm(train_query, ord=2, axis=1)/np.linalg.norm(train_title, ord=2, axis=1)
    train["w2v_sim_cos_query_descr"] = np.sum(train_query*train_descr, axis=1)/np.linalg.norm(train_query, ord=2, axis=1)/np.linalg.norm(train_descr, ord=2, axis=1)

    train["w2v_sim_cos_nonorm_query_title"] = np.sum(train_query*train_title, axis=1)
    train["w2v_sim_cos_nonorm_query_descr"] = np.sum(train_query*train_descr, axis=1)
    train["w2v_sim_cos_skewed_query_title"] = np.sum(train_query*train_title, axis=1)/np.linalg.norm(train_query, ord=2, axis=1)/np.linalg.norm(train_query, ord=2, axis=1)
    train["w2v_sim_cos_skewed_query_descr"] = np.sum(train_query*train_descr, axis=1)/np.linalg.norm(train_query, ord=2, axis=1)/np.linalg.norm(train_query, ord=2, axis=1)



    test["w2v_sim_l1_query_title"] = np.linalg.norm(test_query - test_title, ord=1, axis=1)
    test["w2v_sim_l1_query_descr"] = np.linalg.norm(test_query - test_descr, ord=1, axis=1)
    test["w2v_sim_l2_query_title"] = np.linalg.norm(test_query - test_title, ord=2, axis=1)
    test["w2v_sim_l2_query_descr"] = np.linalg.norm(test_query - test_descr, ord=2, axis=1)
    test["w2v_sim_cos_query_title"] = np.sum(test_query*test_title, axis=1)/np.linalg.norm(test_query, ord=2, axis=1)/np.linalg.norm(test_title, ord=2, axis=1)
    test["w2v_sim_cos_query_descr"] = np.sum(test_query*test_descr, axis=1)/np.linalg.norm(test_query, ord=2, axis=1)/np.linalg.norm(test_descr, ord=2, axis=1)

    test["w2v_sim_cos_nonorm_query_title"] = np.sum(test_query*test_title, axis=1)
    test["w2v_sim_cos_nonorm_query_descr"] = np.sum(test_query*test_descr, axis=1)
    test["w2v_sim_cos_skewed_query_title"] = np.sum(test_query*test_title, axis=1)/np.linalg.norm(test_query, ord=2, axis=1)/np.linalg.norm(test_query, ord=2, axis=1)
    test["w2v_sim_cos_skewed_query_descr"] = np.sum(test_query*test_descr, axis=1)/np.linalg.norm(test_query, ord=2, axis=1)/np.linalg.norm(test_query, ord=2, axis=1)

    return train, test


def to_csr(matrix):
    return scipy.sparse.csr_matrix(matrix)

def add_tfidf_sim_features(train, test):
    train_query, test_query = tfidf.load("query")

    train_title, test_title = tfidf.load("title")
    train["tfidf_sim_cos_query_title"] = to_csr(train_query.multiply(train_title).sum(axis=1)).todense()
    test["tfidf_sim_cos_query_title"] = to_csr(test_query.multiply(test_title).sum(axis=1)).todense()
    del train_title, test_title

    train_descr, test_descr = tfidf.load("descr")
    train["tfidf_sim_cos_query_descr"] = to_csr(train_query.multiply(train_descr).sum(axis=1)).todense()
    test["tfidf_sim_cos_query_descr"] = to_csr(test_query.multiply(test_descr).sum(axis=1)).todense()
    del train_descr, test_descr

    train_title_descr, test_title_descr = tfidf.load("title_descr")
    train["tfidf_sim_cos_query_title_descr"] = to_csr(train_query.multiply(train_title_descr).sum(axis=1)).todense()
    test["tfidf_sim_cos_query_title_descr"] = to_csr(test_query.multiply(test_title_descr).sum(axis=1)).todense()
    del train_title_descr, test_title_descr

    train_attr, test_attr = tfidf.load("attr")
    train["tfidf_sim_cos_query_attr"] = to_csr(train_query.multiply(train_attr).sum(axis=1)).todense()
    test["tfidf_sim_cos_query_attr"] = to_csr(test_query.multiply(test_attr).sum(axis=1)).todense()
    del train_attr, test_attr

    return train, test

def extract_features(data):
    #data = add_features(data)

    columns = ["len_query", "len_title", "len_description", #"len_attr",
               "num_words_query", "num_words_title", "num_words_description", #"num_words_attr",
               "words_in_title", "words_in_description", "words_in_product_info",
               "ratio_title", "ratio_description", "ratio_product_info",
               "counter_nums_title", "counter_nums_description", "counter_nums_query",
               "counter_equal_nums_title_query", "counter_equal_nums_description_query",
               "ratio_equal_nums_title_query", "ratio_equal_nums_description_query",
               "product_has_brand", "counter_query_brand", "counter_query_product_brand",
               "last_word_query_in_title", "last_word_title_in_query", "last_word_query_in_desc",
               "query_in_title", "query_in_description",
               "exact_match_query_title_1_3", "exact_match_query_title_2_3", "exact_match_query_title_2_2",
               "exact_match_query_title_3_3",
               "exact_match_query_descr_2_2", "exact_match_query_descr_2_3", #"exact_match_query_descr_1_3",
               "exact_match_query_descr_3_3",
               "query_in_attr", "query_in_all", #"query_in_title_descr", "query_in_title_attr", "query_in_descr_attr",
               "words_in_attr", "words_in_all", "words_in_title_descr", #"words_in_title_attr", "words_in_descr_attr",
               "ratio_attr", "ratio_all", "ratio_title_descr", "ratio_title_attr", "ratio_descr_attr",
               "w2v_sim_l1_query_title", "w2v_sim_l1_query_descr", "w2v_sim_l2_query_title",
               "w2v_sim_l2_query_descr", "w2v_sim_cos_query_title", "w2v_sim_cos_query_descr",
               "w2v_sim_cos_skewed_query_title", "w2v_sim_cos_skewed_query_descr",
               "tfidf_sim_cos_query_title", "tfidf_sim_cos_query_descr", "tfidf_sim_cos_query_title_descr",
               "tfidf_sim_cos_query_attr",
               #"w2v_sim_cos_nonorm_query_title", #"w2v_sim_cos_nonorm_query_descr",
               #"w2v_sim_cos_query_title", "w2v_sim_cos_query_descr", "w2v_sim_l1_query_title"

               "query_color_counter", "title_color_counter",# "description_color_counter", "info_color_counter",
               "colors_in_title", "colors_in_colors", "colors_in_description", "color_color_counter"]
    data_ = data[columns].copy(deep=True)
    del data
    return data_


def sizes_to_standart(text):
    #inch
    text = re.sub(r"([0-9]) in\.", r"\1inch", text)
    text = re.sub(r"([0-9])in\.", r"\1inch", text)
    text = re.sub(r"([0-9]) inch", r"\1inch", text)
    text = re.sub(r"([0-9]) inche", r"\1inch", text)
    text = re.sub(r"([0-9])inche", r"\1inch", text)

    """
    #feet
    text = re.sub(r"([0-9]) ft\.", r"\1feet", text)
    text = re.sub(r"([0-9])ft\.", r"\1feet", text)
    text = re.sub(r"([0-9]) feet", r"\1feet", text)
    text = re.sub(r"([0-9]) foot", r"\1feet", text)
    text = re.sub(r"([0-9])foot", r"\1feet", text)

    #pound
    text = re.sub(r"([0-9]) lb\.", r"\1pound", text)
    text = re.sub(r"([0-9])lb\.", r"\1pound", text)
    text = re.sub(r"([0-9]) pound", r"\1pound", text)
    text = re.sub(r"([0-9]) pounds", r"\1pound", text)

    #volt
    text = re.sub(r"([0-9]) v\.", r"\1volt", text)
    text = re.sub(r"([0-9])v\.", r"\1volt", text)
    text = re.sub(r"([0-9]) v ", r"\1volt ", text)
    text = re.sub(r"([0-9])v ", r"\1volt ", text)
    text = re.sub(r"([0-9]) volt", r"\1volt", text)
    text = re.sub(r"([0-9]) volts", r"\1volt", text)
    """

    return text


def stem(text):
    text = text.lower()
    re.sub(r"-", r" ", text)
    text = sizes_to_standart(text)

    text = " ".join([stemmer.stem(word) for word in nltk.tokenize.word_tokenize(text)])
    return text


def count_text1_in_text2(text1, text2):
    words = nltk.tokenize.word_tokenize(text1)
    counter = 0

    for word in words:
        if text2.find(word) >= 0:
            counter += 1

    return counter


def count_list_in_text(list_, text_):
    result = 0
    text = " " + re.sub(r"[^a-z\-\&]", " ", text_) + " "

    for element in list_:
        if text.find(" " + element + " ") >= 0:
            result += 1

    return result


def extract_nums(text_):
    text = re.sub(r"[^0-9\.]", r" ", text_)
    text = re.sub(r"([^0-9])\.+", r"\1 ", text)
    text = re.sub(r"\.([^0-9])+", r"\1 ", text)

    list1 = text.split()
    list2 = []
    for element in list1:
        if element != ".":
            list2.append(element)

    return np.array(list2, dtype=str)


def count_equal_nums(array1, array2):
    counter = 0
    for num2 in array2:
        for num1 in array1:
            if num2 == num1:
                counter += 1
                break

    return counter


def add_features_exact_match(data):
    data["exact_match_query_title_1_3"] = data.apply(lambda row: add_ngram_query_exact_match(row["search_term"],
                                                                                             row["product_title"],
                                                                                             1, 3),
                                                     axis=1)
    data["exact_match_query_title_2_3"] = data.apply(lambda row: add_ngram_query_exact_match(row["search_term"],
                                                                                             row["product_title"],
                                                                                             2, 3),
                                                     axis=1)
    data["exact_match_query_title_2_2"] = data.apply(lambda row: add_ngram_query_exact_match(row["search_term"],
                                                                                             row["product_title"],
                                                                                             2, 2),
                                                     axis=1)
    data["exact_match_query_title_3_3"] = data.apply(lambda row: add_ngram_query_exact_match(row["search_term"],
                                                                                             row["product_title"],
                                                                                             3, 3),
                                                     axis=1)

    data["exact_match_query_descr_1_3"] = data.apply(lambda row: add_ngram_query_exact_match(row["search_term"],
                                                                                             row["product_description"],
                                                                                             1, 3),
                                                     axis=1)
    data["exact_match_query_descr_2_3"] = data.apply(lambda row: add_ngram_query_exact_match(row["search_term"],
                                                                                             row["product_description"],
                                                                                             2, 3),
                                                     axis=1)
    data["exact_match_query_descr_2_2"] = data.apply(lambda row: add_ngram_query_exact_match(row["search_term"],
                                                                                             row["product_description"],
                                                                                             2, 2),
                                                     axis=1)
    data["exact_match_query_descr_3_3"] = data.apply(lambda row: add_ngram_query_exact_match(row["search_term"],
                                                                                             row["product_description"],
                                                                                             3, 3),
                                                     axis=1)
    return data


def add_counter_features(data):
    data["words_in_title"] = data["query_product"].map(lambda x: count_text1_in_text2(x.split("\t")[0],
                                                                                      x.split("\t")[1]))
    data["words_in_description"] = data["query_product"].map(lambda x: count_text1_in_text2(x.split("\t")[0],
                                                                                            x.split("\t")[2]))
    data["words_in_attr"] = data.apply(lambda row: count_text1_in_text2(row["search_term"],
                                                                        row["attr"]),
                                       axis=1)
    data["words_in_all"] = data.apply(lambda row: count_text1_in_text2(row["search_term"],
                                                                       row["product_title"]+row["product_description"]+row["attr"]),
                                      axis=1)
    data["words_in_title_descr"] = data.apply(lambda row: count_text1_in_text2(row["search_term"],
                                                                               row["product_title"]+row["product_description"]),
                                              axis=1)
    data["words_in_title_attr"] = data.apply(lambda row: count_text1_in_text2(row["search_term"],
                                                                              row["product_title"]+row["attr"]),
                                             axis=1)
    data["words_in_descr_attr"] = data.apply(lambda row: count_text1_in_text2(row["search_term"],
                                                                              row["product_description"]+row["attr"]),
                                             axis=1)
    data["words_in_product_info"] = data["words_in_title"] + data["words_in_description"]


    data["ratio_title"] = data["words_in_title"]/data["len_query"]
    data["ratio_description"] = data["words_in_description"]/data["len_query"]
    data["ratio_attr"] = data["words_in_attr"]/data["len_query"]
    data["ratio_product_info"] = data["words_in_product_info"]/data["len_query"]
    data["ratio_all"] = data["words_in_all"]/data["len_query"]
    data["ratio_title_descr"] = data["words_in_title_descr"]/data["len_query"]
    data["ratio_title_attr"] = data["words_in_title_attr"]/data["len_query"]
    data["ratio_descr_attr"] = data["words_in_descr_attr"]/data["len_query"]

    return data


def add_length_features(data):
    data["len_query"] = data["search_term"].map(lambda x: len(x))
    data["len_title"] = data["product_title"].map(lambda x: len(x))
    data["len_description"] = data["product_description"].map(lambda x: len(x))
    data["len_attr"] = data["attr"].map(lambda x: len(x))

    data["num_words_query"] = data["search_term"].map(lambda x: len(nltk.tokenize.word_tokenize(x)))
    data["num_words_title"] = data["product_title"].map(lambda x: len(nltk.tokenize.word_tokenize(x)))
    data["num_words_description"] = data["product_description"].map(lambda x: \
                                    len(nltk.tokenize.word_tokenize(x)))
    data["num_words_attr"] = data["attr"].map(lambda x: len(nltk.tokenize.word_tokenize(x)))

    return data


def add_features_whole_word(data):
    data["query_in_title"] = data["query_product"].apply(lambda x: str_whole_word(x.split('\t')[0],
                                                                                  x.split('\t')[1],
                                                                                  0))
    data["query_in_description"] = data["query_product"].apply(lambda x: str_whole_word(x.split('\t')[0],
                                                                                        x.split('\t')[2],
                                                                                        0))
    data["query_in_attr"] = data.apply(lambda row: str_whole_word(row["search_term"],
                                                                  row["attr"], 0),
                                      axis=1)
    data["query_in_all"] = data.apply(lambda row: str_whole_word(row["search_term"],
                                                                 row["product_title"]+row["product_description"]+row["attr"], 0),
                                      axis=1)
    data["query_in_title_descr"] = data.apply(lambda row: str_whole_word(row["search_term"],
                                                                         row["product_title"]+row["product_description"], 0),
                                      axis=1)
    data["query_in_title_attr"] = data.apply(lambda row: str_whole_word(row["search_term"],
                                                                        row["product_title"]+row["attr"], 0),
                                      axis=1)
    data["query_in_descr_attr"] = data.apply(lambda row: str_whole_word(row["search_term"],
                                                                        row["product_description"]+row["attr"], 0),
                                      axis=1)
    return data


def add_features_last_word(data):
    data["last_word_title_in_query"] = data.apply(lambda row: last_word_in(row["search_term"],
                                                                           row["product_title"]),
                                                  axis=1)
    data["last_word_query_in_title"] = data.apply(lambda row: last_word_in(row["product_title"],
                                                                           row["search_term"]),
                                                  axis=1)
    data["last_word_query_in_desc"] = data.apply(lambda row: last_word_in(row["product_description"],
                                                                          row["search_term"]),
                                                 axis=1)
    return data


def add_features_brands(data, brands):
    data["product_has_brand"] = data["brand"].apply(lambda x: int(not(x is np.nan)))
    data["counter_query_brand"] = data["search_term"].apply(lambda x: count_list_in_text(brands, x.lower()))
    data["counter_query_product_brand"] = data.apply(lambda row: count_list_in_text(str(row["brand"]).split(","),
                                                                                    row["search_term"].lower()),
                                                     axis=1)

    return data


def add_features_numbers(data):
    data["nums_title"] = data["product_title"].apply(lambda x: extract_nums(x))
    data["nums_description"] = data["product_description"].apply(lambda x: extract_nums(x))
    data["nums_query"] = data["search_term"].apply(lambda x: extract_nums(x))

    data["counter_nums_title"] = data["nums_title"].apply(lambda x: len(x))
    data["counter_nums_description"] = data["nums_description"].apply(lambda x: len(x))
    data["counter_nums_query"] = data["nums_query"].apply(lambda x: len(x))

    data["counter_equal_nums_title_query"] = data.apply(lambda row: count_equal_nums(row['nums_title'],
                                                                                     row['nums_query']),
                                                        axis=1)
    data["counter_equal_nums_description_query"] = data.apply(lambda row: count_equal_nums(row['nums_description'],
                                                                                           row['nums_query']),
                                                              axis=1)

    data["ratio_equal_nums_title_query"] = (data["counter_equal_nums_title_query"] + 1)/(data["counter_nums_query"] + 1)
    data["ratio_equal_nums_description_query"] = (data["counter_equal_nums_description_query"] + 1)/(data["counter_nums_query"] + 1)

    return data


def preprocess(data, brands):
    data = add_features_numbers(data)
    data = add_features_brands(data, brands)

    data["product_title_raw"] = data["product_title"]
    data["product_description_raw"] = data["product_description"]
    data["attr_raw"] = data["attr"]
    data["search_term_raw"] = data["search_term"]

    data["product_title"] = data["product_title"].apply(lambda x: stem(x))
    data["product_description"] = data["product_description"].apply(lambda x: stem(x))
    data["attr"] = data["attr"].apply(lambda x: stem(x))
    data["search_term"] = data["search_term"].apply(lambda x: stem(x))
    data["query_product"] = data["search_term"] + "\t" + data["product_title"] +\
                            "\t"+data["product_description"]

    data = add_features_last_word(data)
    data = add_features_whole_word(data)
    data = add_features_exact_match(data)
    data = add_length_features(data)
    data = add_counter_features(data)

    """
    data["query_colors"] = data["search_term"].apply(lambda x: extract_color(x, colors)).astype("str")
    data["title_colors"] = data["product_title"].apply(lambda x: extract_color(x, colors)).astype("str")
    data["description_colors"] = data["product_description"].apply(lambda x: extract_color(x, colors)).astype("str")

    data["query_product_colors"] = (data["query_colors"] + "\t" + data["title_colors"] +\
                                   "\t"+data["description_colors"] + "\t" + data["colors"]).astype("str")

    data["colors_in_title"] = data["query_product_colors"].apply(lambda x: count_text1_in_text2(x.split("\t")[0],
                                                                                              x.split("\t")[1]))
    data["colors_in_description"] = data["query_product_colors"].apply(lambda x: count_text1_in_text2(x.split("\t")[0],
                                                                                                    x.split("\t")[2]))
    data["colors_in_colors"] = data["query_product_colors"].apply(lambda x: count_text1_in_text2(x.split("\t")[0],
                                                                                               x.split("\t")[3]))
    data["colors_in_product_info"] = data["colors_in_title"] + data["colors_in_description"]

    data["query_color_counter"] = data["query_colors"].apply(lambda x: len(nltk.tokenize.word_tokenize(x)))
    data["title_color_counter"] = data["title_colors"].apply(lambda x: len(nltk.tokenize.word_tokenize(x)))
    data["description_color_counter"] = data["description_colors"].apply(lambda x: len(nltk.tokenize.word_tokenize(x)))
    data["info_color_counter"] = data["title_color_counter"] + data["description_color_counter"]
    data["color_color_counter"] = data["colors"].apply(lambda x: len(nltk.tokenize.word_tokenize(x)))
    """

    return data


"""
def get_color(attributes_):
    attributes = attributes_[attributes_["name"].apply(lambda x: x.find("color")) >= 0].values
    color_df = pd.DataFrame()
    color_set = set()

    product = attributes[0, 0]
    list_colors = []

    for i in range(len(attributes)):
        if attributes[i, 0] == product:
            list_colors.append(attributes[i, 2])
        else:
            color_df = color_df.append(pd.DataFrame([[product, " ".join(list_colors)]],
                                                    columns=["product_uid", "colors"]))
            color_set.update(list_colors)
            product = attributes[i, 0]
            list_colors = [attributes[i, 2]]

    color_df = color_df.append(pd.DataFrame([[product, " ".join(list_colors)]],
                                            columns=["product_uid", "colors"]))

    color_set = np.array(color_set)
    return color_df, color_set
"""


def make_typos(text_):
    brand_set = set([text_.lower()])

    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text_)
    brand_set.add(text_.lower())

    text = re.sub(r"-", r" ", text_)
    brand_set.add(text.lower())

    text = re.sub(r"-", r"", text_)
    brand_set.add(text.lower())

    text = re.sub(r" ", r"", text_)
    brand_set.add(text.lower())

    text = re.sub(r"'", r"", text_)
    brand_set.add(text.lower())

    text = re.sub(r" & ", r" ", text_)
    brand_set.add(text.lower())

    return brand_set


def add_brands(train, test, attributes):
    brand_df = pd.DataFrame()
    brands = set()

    values = attributes[attributes["name"] == "MFG Brand Name"]["value"].values
    product_uids = attributes[attributes["name"] == "MFG Brand Name"]["product_uid"].values

    for val, uid in zip(values, product_uids):
        val_with_typos = make_typos(val)
        brand_df = brand_df.append(pd.DataFrame([[uid, ",".join(val_with_typos)]],
                                                  columns=["product_uid", "brand"]))
        brands.update(val_with_typos)

    train = pd.merge(train, brand_df, how="left", on="product_uid")
    test = pd.merge(test, brand_df, how="left", on="product_uid")
    train["brand"] = train["brand"].astype(str)
    test["brand"] = test["brand"].astype(str)

    brand_df.to_pickle("cache/brand_df")

    brands = np.array(brands)
    brands.dump("cache/brands")

    return train, test, np.array(list(brands.tolist()))


def generate_tfidf(train, test):
    text_train = train["search_term"].values + train["product_title"].values + train["product_description"].values
    text_test = test["search_term"].values + test["product_title"].values + test["product_description"].values

    tfidf1 = sklearn.feature_extraction.text.TfidfVectorizer(analyzer="word", ngram_range=(1, 3),
                                                             min_df=25)
    text_train_tfidf = tfidf1.fit_transform(text_train)
    text_test_tfidf = tfidf1.transform(text_test)

    return text_train_tfidf, text_test_tfidf


def add_attr(train, test, attr_df_):
    attr_df = attr_df_.copy(deep=True)
    attr_df = attr_df.dropna()
    attr_df = attr_df.reset_index(drop=True)

    d = {}
    for i in range(len(attr_df)):
        if str(int(attr_df.product_uid[i])) in d:
            d[str(int(attr_df.product_uid[i]))][1] += " " + str(attr_df['value'][i]).replace('\t'," ")
        else:
            d[str(int(attr_df.product_uid[i]))] = [int(attr_df.product_uid[i]), str(attr_df['value'][i])]

    attr_df = pd.DataFrame.from_dict(d, orient='index')
    attr_df.columns = ["product_uid", "attr"]

    train = pd.merge(train, attr_df, how="left", on="product_uid")
    test = pd.merge(test, attr_df, how="left", on="product_uid")
    train["attr"] = train["attr"].astype(str)
    test["attr"] = test["attr"].astype(str)

    attr_df.to_pickle("cache/attr_df")

    return train, test


def preprocess_data(train_, test_, attributes_, load=True):
    if load:
        train = pd.read_pickle("cache/df_train")
        test = pd.read_pickle("cache/df_test")

        #brand_df = pd.read_pickle("cache/brand_df")
        #brands = np.array(list(np.load("cache/brands").tolist()))

        #train = add_features_brands(train, brands)
        #test = add_features_brands(test, brands)
        #train, test = add_attr(train, test, attributes_)

        """
        names = ["w2v_sim_cos_query_title", "w2v_sim_cos_query_descr",
                 "w2v_sim_cos_skewed_query_title", "w2v_sim_cos_skewed_query_descr"]

        for name in names:
            train[name][train.isnull()[name]] = 0
            test[name][test.isnull()[name]] = 0
        """

        train = add_color_features(train)
        test = add_color_features(test)

        #train, test = add_w2v_sim_features(train, test)

        #color_df = pd.read_pickle("cache/color_df")
        #color_set = np.load("cache/color_set")

        #color_set = make_color_set(color_set)


        train.to_pickle("cache/df_train")
        test.to_pickle("cache/df_test")

    else:
        train = train_.copy(deep=True)
        test = test_.copy(deep=True)
        #attributes = attributes_.copy(deep=True)
        #attributes = preprocess_attributes(attributes)

        #color_df, color_set = get_color(attributes)
        #color_set.dump("cache/color_set")
        #color_set = make_color_set(color_set)

        #train = pd.merge(train, color_df, how="left", on="product_uid")
        #test = pd.merge(test, color_df, how="left", on="product_uid")
        #train["colors"] = train["colors"].astype("str")
        #test["colors"] = test["colors"].astype("str")

        train, test, brands = add_brands(train, test, attributes_)
        train, test = add_attr(train, test, attributes_)


        train = preprocess(train, brands)
        test = preprocess(test, brands)

        train, test = add_w2v_sim_features(train, test)
        train, test = add_tfidf_sim_features(train, test)

        train.to_pickle("cache/df_train")
        test.to_pickle("cache/df_test")
        #color_df.to_pickle("cache/color_df")
        #color_set.dump("cache/color_set")

    return train, test
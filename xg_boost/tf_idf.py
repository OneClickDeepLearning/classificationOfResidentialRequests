import collections
import csv
import math
import pickle


def read_pickle(file_path):
    data = ""
    if not file_path.endswith(".pickle"):
        print("[ERROR] file suffix missing or file suffix is not appropriate.")
    else:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    return data


def read_data():
    data_lst = []
    try:
        with open('4w_trainset.csv', 'r', encoding='GB18030') as db01:
            reader = csv.reader(db01)
            for row in reader:
                data_lst.append(row)
        return data_lst
    except csv.Error as e:
        print(e)


def cal_tfidf():
    dict_w4 = read_pickle("tfidf_4w_training_set.pickle")
    data_idf = read_pickle("tf_idf_idf_4w_training_set.pickle")

    total_amount = sum(dict_w4.values())
    print(total_amount)
    # tf
    dict_new = collections.defaultdict(list)
    for key, value in dict_w4.items():
        word_freq = value / total_amount
        # tf
        dict_new[key].append(word_freq)
        # idf
        x = total_amount / len(data_idf[key])
        # tfidf
        tfidf_value = math.log(x + 1) * word_freq

        dict_new[key].append(tfidf_value)
    sorted_dict = sorted(dict_new.items(), key=lambda k: k[1][1])

    sort_tfidf_value = {}
    for value in sorted_dict:
        sort_tfidf_value[value[0]] = value[1][1]

    print(sort_tfidf_value)
    # Normalization
    max_scores = max(sort_tfidf_value.values())
    min_scores = min(sort_tfidf_value.values())
    for key in sort_tfidf_value.keys():
        x = sort_tfidf_value[key]
        sort_tfidf_value[key] = (x - min_scores) / (max_scores - min_scores)
    with open('tf_idf_value.pickle', 'wb') as f:
        pickle.dump(sort_tfidf_value, f, pickle.HIGHEST_PROTOCOL)

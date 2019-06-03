import math
import pickle
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from sklearn.cluster import DBSCAN
from sklearn.cluster.optics_ import OPTICS
from sklearn.decomposition import PCA

from nlp import jieba


def preprocess():
    regexp_punct = re.compile("^[\s+\!\/_,$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）. 〉《 》〔〕；～ ％]")
    industry_lst = []
    industry_lda = []
    industry_lda_dict = {}

    with open('origin_class_label.txt') as file:
        for iid, row in enumerate(file):
            row = row.strip('\n').split(',')
            temp = re.sub(regexp_punct, " ", row[0])
            token = jieba.cut(temp)
            industry_lst.extend(i for i in token if len(i) > 1)
            industry_lda.append(industry_lst)
            industry_lda_dict[iid] = temp

    for iid, i in enumerate(token):
        temp2 = [k for k in jieba.cut(i) if len(k) > 1]
        industry_lda.append(temp2)
        industry_lda_dict[iid] = i

    industry_lst = list(set(industry_lst))

    with open('agency_to_vec.pickle', 'rb') as file:
        place_w2v = pickle.load(file)

    token = list(place_w2v.keys())

    w2v = [value for value in place_w2v.values]
    pca = PCA(n_components=2)
    w2v_2d_array = np.array(pca.fit_transform(w2v))
    token_dict = {token[i]: v for i, v in enumerate(w2v_2d_array)}

    return industry_lst, w2v_2d_array, token_dict, industry_lda


def run_dbscan(w2v_2d_array):
    db = DBSCAN(eps=0.8, min_samples=3).fit_predict(X)
    plt.scatter(w2v_2d_array[:, 0], w2v_2d_array[:, 1], c=db)
    plt.show()


def run_optics(w2v_2d_array, industry_lst):
    clust = OPTICS(min_samples=1, rejection_ratio=0.7)
    opt = clust.fit_predict(w2v_2d_array)
    labels = clust.labels_
    optics_dict = defaultdict(list)
    for i, v in enumerate(labels):
        optics_dict[v].append(industry_lst[i])

    max_k = max(opt)
    for k in range(0, max_k):
        Xk = w2v_2d_array[clust.labels_ == k]
        plt.scatter(Xk[:, 0], Xk[:, 1], label=k)
        plt.legend()
    plt.title('OPTICS Clustering\n'
              'min_samples= 4, rejection_ratio=0.67')

    plt.scatter(w2v_2d_array[clust.labels_ == -1, 0], w2v_2d_array[clust.labels_ == -1, 1], c='black', marker='+')
    plt.show()
    return max_k


def cal_sim(vector1, vector2):
    op1 = np.sqrt(np.sum(np.square(vector1 - vector2)))
    return op1


# LDA
def run_lda_with_entropy(industry_lda, token_dict, max_k=5):
    common_dictionary = corpora.Dictionary(industry_lda)
    common_corpus = [common_dictionary.doc2bow(text) for text in industry_lda]
    ldamodel = LdaModel(corpus=common_corpus, num_topics=max_k + 1, id2word=common_dictionary)
    result = ldamodel.print_topics(num_topics=max_k + 1, num_words=10)
    center_lst = []
    for i in range(max_k + 1):
        result2 = ldamodel.get_topic_terms(topicid=i)
        sum_word = 0
        center = 0
        length = len(result2)
        for v in result2:
            if common_dictionary[v[0]] in token_dict.keys():
                center += token_dict[common_dictionary[v[0]]]
        center_lst.append(center / length)

    industry_with_center_distance = []
    sum_temp5_lst = []
    for i in industry_lda:
        temp2 = []
        for k in i:
            temp = []
            if k in token_dict.keys():
                for j in center_lst:
                    temp.append((cal_sim(np.array(token_dict[k]), j)))
            if len(temp) > 0:
                temp2.append(temp)
        if len(temp2) > 0:
            temp3 = np.array(temp2)
            temp4 = np.mean(temp3, axis=0)
            temp5 = np.sum(temp3)
        else:
            temp4 = []
            for i in range(0, max_k + 1):
                temp4.append(0.0)
            temp5 = temp4

        industry_with_center_distance.append(temp4)
        sum_temp5_lst.append(temp5)

    entro_result_final = {}

    for number, i in enumerate(industry_lda):
        entro_result_2 = []
        for k in i:
            entro_result = []
            if k in token_dict.keys():
                for j in center_lst:
                    temp = cal_sim(np.array(token_dict[k]), j)
                    temp_value = temp / sum_temp5_lst[number]
                    entro_result.append(temp_value * math.log(temp_value))
            entro_result_2.append(entro_result)
        if len(entro_result_2) > 0:
            temp5 = np.zeros(shape=(1, max_k + 1), dtype=float)
            for w in entro_result_2:
                if len(w) > 0:
                    temp4 = np.array(w)
                    temp5 += temp4

            list_temp5 = list(temp5[0])
            entro_result_final[number] = list_temp5.index(max(list_temp5))

    final_result = {}
    for i in range(0, max_k + 1):
        for key, value in entro_result_final.items():
            if value == i:
                final_result[i].append(industry_lda[key])


def main():
    industry_lst, w2v_2d_array, token_dict, industry_lda = preprocess()
    run_dbscan(w2v_2d_array)
    max_k = run_optics(w2v_2d_array, industry_lst)
    run_lda_with_entropy(industry_lda, token_dict, max_k=max_k)


if __name__ == '__main__':
    main()

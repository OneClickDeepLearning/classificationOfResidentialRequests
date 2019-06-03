import csv

import matplotlib.pyplot as plt
from sklearn import decomposition

from feature_engineering.feature_anlysis import data


def read_data():
    dict_agency_label = {}
    with open('origin_class_label.txt', 'r') as f:
        for row in f:
            temp = row.split(',')
            dict_agency_label[temp[0]] = temp[1].strip('\n')

    data_lst = []
    try:
        with open('4w_trainset.csv', 'r', encoding='GB18030') as db01:
            reader = csv.reader(db01)
            for row in reader:
                data_lst.append(row)
        return data_lst, dict_agency_label
    except csv.Error as e:
        print(e)


def count_label(data_lst):
    y_label = []
    for i in data_lst[1:]:
        y_label.append(int(i[9]))

    count_zero = 0
    count_one = 0
    count_two = 0
    count_three = 0
    count_four = 0

    for i in y_label:
        if i == 0:
            count_zero += 1
        if i == 1:
            count_one += 1
        if i == 2:
            count_two += 1
        if i == 3:
            count_three += 1
        if i == 4:
            count_four += 1
    print('第0类的有{}个'.format(count_zero))
    print('第1类的有{}个'.format(count_one))
    print('第2类的有{}个'.format(count_two))
    print('第3类的有{}个'.format(count_three))
    print('第4类的有{}个'.format(count_four))


def draw_pic(dict_agency_label):
    agency_2_vec = data.read_pickle('agency_2_vec_testset1.pickle')
    vec_lst = []
    for k, v in agency_2_vec.items():
        vec_lst.append(v)
    pca = decomposition.PCA(n_components=2)
    newData = pca.fit_transform(vec_lst)
    agency_2d_vec = {}
    for i, k in enumerate(agency_2_vec.keys()):
        agency_2d_vec[k] = newData[i]

    colors = ['b', 'c', 'y', 'm', 'r']

    plot_zero = []
    plot_one = []
    plot_two = []
    plot_three = []
    plot_four = []
    for k, v in dict_agency_label.items():
        if v == '0' and k in agency_2d_vec.keys():
            plot_zero.append(agency_2d_vec[k])
        if v == '1' and k in agency_2d_vec.keys():
            plot_one.append(agency_2d_vec[k])
        if v == '2' and k in agency_2d_vec.keys():
            plot_two.append(agency_2d_vec[k])
        if v == '3' and k in agency_2d_vec.keys():
            plot_three.append(agency_2d_vec[k])
        if v == '4' and k in agency_2d_vec.keys():
            plot_four.append(agency_2d_vec[k])

    plotx0 = []
    ploty0 = []
    plotx1 = []
    ploty1 = []
    plotx2 = []
    ploty2 = []
    plotx3 = []
    ploty3 = []
    plotx4 = []
    ploty4 = []

    # lanse di 0 lei
    for i in range(len(plot_zero)):
        plotx0.append(plot_zero[i][0])
        ploty0.append(plot_zero[i][1])
        # plt.scatter(plot_zero[i][0],plot_zero[i][1],color = colors[0], marker = 'o')
    # lv se di 1 lei
    for i in range(len(plot_one)):
        plotx1.append(plot_one[i][0])
        ploty1.append(plot_one[i][1])
        # plt.scatter(plot_one[i][0],plot_one[i][1],color = colors[1], marker = 'v')
    # hong se di 1 lei
    for i in range(len(plot_two)):
        plotx2.append(plot_two[i][0])
        ploty2.append(plot_two[i][1])
        # plt.scatter(plot_two[i][0],plot_two[i][1],color = colors[2], marker = '^')
    # hei se di 2 lei
    for i in range(len(plot_three)):
        plotx3.append(plot_three[i][0])
        ploty3.append(plot_three[i][1])
        # plt.scatter(plot_three[i][0],plot_three[i][1],color = colors[3], marker = '*')
    # qian lan se di 3 lei
    for i in range(len(plot_four)):
        plotx4.append(plot_four[i][0])
        ploty4.append(plot_four[i][1])
        # plt.scatter(plot_four[i][0],plot_four[i][1],color = colors[4], marker = 'x')

    plt.scatter(plotx0, ploty0, marker='x', color=colors[0], label='Class Zero')
    plt.scatter(plotx1, ploty1, marker='o', color=colors[1], label='Class One')
    plt.scatter(plotx2, ploty2, marker='v', color=colors[2], label='Class Two')
    plt.scatter(plotx3, ploty3, marker='^', color=colors[3], label='Class Three')
    plt.scatter(plotx4, ploty4, marker='*', color=colors[4], label='Class Four')
    plt.xlabel('2D X Axis')
    plt.xticks([])
    plt.ylabel('2D Y Axis')
    plt.yticks([])
    plt.legend()
    plt.show()


def main():
    data_lst, dict_agency_label = read_data()
    count_label(data_lst)
    draw_pic(dict_agency_label)

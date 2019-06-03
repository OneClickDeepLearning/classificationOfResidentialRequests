import math

from feature_engineering.feature_anlysis import data

# four-related class level with count
FILE_NAME_1 = "related_class_level_with_count.pickle"
# responsible departments with four-related class
FILE_NAME_2 = "responsible_departments_with_count.pickle"


def cal_information_value():
    lst_industry = data.read_pickle(FILE_NAME_1)
    dict_industry = data.read_pickle(FILE_NAME_2)
    total_1, total_2, total_3 = 0, 0, 0

    for key, value in dict_industry.items():
        for i in value:
            for key_2, value_2 in i.items():
                if key_2 in lst_industry.keys():
                    value_2.append(
                        (lst_industry.get(key_2) - value_2[0]) if (lst_industry.get(key_2) - value_2[0]) > 0 else 1)
                    value_2.append(lst_industry.get(key_2))
                    value_2.append(value_2[0] / value_2[2])
                    total_1 += value_2[0]
                    total_2 += value_2[1]
                    total_3 += value_2[2]

    for key, value in dict_industry.items():
        for i in value:
            for key_2, value_2 in i.items():
                if key_2 in lst_industry.keys():
                    # compute WOE
                    if value_2[1] == 0:
                        value_2.append(0)
                        value_2.append(0)
                    else:
                        # compute WOE and IV
                        x = (value_2[0] / value_2[1]) / (total_1 / total_2)
                        value_2.append(math.log(x))
                        temp_IV = ((value_2[0] / total_1) - (value_2[1] / total_2)) * value_2[4]
                        value_2.append(temp_IV)
    return lst_industry, dict_industry


def analysis_res(lst_industry, dict_industry):
    dict_none_final = {}
    dict1_final = {}
    dict2_final = {}
    dict3_final = {}
    dict4_final = {}
    dict5_final = {}
    dict_res = {k: 0 for k in lst_industry.keys()}

    for key, value in dict_industry.items():
        dict_none = {}
        # <0.02
        dict1 = {}
        # 0.02~0.1
        dict2 = {}
        # 0.1~0.3
        dict3 = {}
        # 0.3~0.5
        dict4 = {}
        # 0.5~
        dict5 = {}
        for i in value:
            for key_2, value_2 in i.items():
                if len(value_2) == 6:
                    if value_2[5] is None:
                        dict_none[key_2] = value_2
                    elif value_2[5] < 0.02:
                        dict1[key_2] = value_2
                    elif (value_2[5] > 0.02) and (value_2[5] < 0.1):
                        dict2[key_2] = value_2
                    elif (value_2[5] > 0.1) and (value_2[5] < 0.3):
                        dict3[key_2] = value_2
                    elif (value_2[5] > 0.3) and (value_2[5] < 0.5):
                        dict4[key_2] = value_2
                    else:
                        dict5[key_2] = value_2

        dict_none_final[key] = dict_none
        for k, v in dict1.items():
            if len(v) > 0:
                dict1_final[key] = dict1
            if k in dict_res.keys():
                dict_res[k] += v[-1]

        for k, v in dict2.items():
            if len(v) > 0:
                dict2_final[key] = dict2
            if k in dict_res.keys():
                dict_res[k] += v[-1]

        for k, v in dict3.items():
            if len(v) > 0:
                dict3_final[key] = dict3
            if k in dict_res.keys():
                dict_res[k] += v[-1]

        for k, v in dict4.items():
            if len(v) > 0:
                dict4_final[key] = dict4
            if k in dict_res.keys():
                dict_res[k] += v[-1]

        for k, v in dict5.items():
            if len(v) > 0:
                dict5_final[key] = dict5
            if k in dict_res.keys():
                dict_res[k] += v[-1]

    print("IV值为无穷的有：")
    lst_temp = [v for v in dict_res.values() if v == 0]
    print(len(lst_temp))

    print("IV值为<0.02的有：")
    lst_temp = [v for v in dict_res.values() if 0 < v < 0.02]
    print(len(lst_temp))

    print("IV值为0.02~0.1的有：")
    lst_temp = [v for v in dict_res.values() if 0.02 < v < 0.1]
    print(len(lst_temp))

    print("IV值为0.1~0.3的有：")
    lst_temp = [v for v in dict_res.values() if 0.1 < v < 0.3]
    print(len(lst_temp))

    print("IV值为0.3~0.5的有：")
    lst_temp = [v for v in dict_res.values() if 0.3 < v < 0.5]
    print(len(lst_temp))

    print("IV值为0.5以上的有：")
    lst_temp = [v for v in dict_res.values() if v > 0.5]
    print(len(lst_temp))


def main():
    lst_industry, dict_industry = cal_information_value()
    analysis_res(lst_industry, dict_industry)


if __name__ == '__main__':
    main()

import csv
import data_cleaning.cleaning as cl

file2 = '/Users/sunjincheng/Documents/valid_data_all.csv'
file2 = open(file2, encoding='gb18030')
all_data = csv.reader(file2)

file = '../data/all_labels.txt'


def count_dept(valid_data, label_file):
    valid_data = open(valid_data, encoding='gb18030')
    all_data = csv.reader(valid_data)
    lines = -1
    pre_line = ''
    pre_dept = ''
    departments = []
    for line in all_data:
        lines += 1
        if lines == 0:
            continue
        department = line[14]
        if department != pre_line:
            pre_dept, _ = cl.hgdProcess_dept(department)
            pre_line = department
        department = pre_dept
        # line[13] = department
        #不能覆盖
        departments.append(department)


    depts = list(set(departments))

    count = 0

    with open(label_file, 'w',encoding='gb18030') as f:
        for dept in depts:
            f.write(dept + ',' + str(count) + '\n')
            count += 1
    return lines, count

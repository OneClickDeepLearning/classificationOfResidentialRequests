import DataCleaning.cleaning as cl
import pandas as pd
import csv
import numpy as np

def valid2set(label_file,valid_data,dataset_file):
    labels = {}
    for line in open(label_file, 'r', encoding='gb18030'):
        labels[line.split(',')[0]] = int(line.split(',')[1])

    valid_data = csv.reader(open(valid_data,encoding='gb18030'))
    dataset = []
    # 645924
    count = -1
    pre_line = ''
    pre_dept = ''
    for line in valid_data:
        count += 1
        if count == 0:
            continue
        ID = line[0]
        job_num = line[1]
        job_cls1 = line[3]
        job_cls2 = line[4]
        job_cls3 = line[5]
        job_cls4 = line[6]
        content = line[8]
        old_dept = line[14]
        department = old_dept

        # process "处置单位"
        if department != pre_line:
            pre_dept, _ = cl.hgdProcess_dept(department)
            pre_line = department
        department = pre_dept
        label = labels[department]

        dataset.append([ID, job_num, job_cls1, job_cls2, job_cls3, job_cls4, content, old_dept, department, label])

        # if count == num:
        #     break

    dataset = np.array(dataset)
    dataset = pd.DataFrame(dataset, index=dataset[:, 0],
                           columns={'ID', '工单编号',
                                    '行业分类1级', '行业分类2级', '行业分类3级', '行业分类4级',
                                    '诉求内容', '原处置单位', '处置单位', '单位类别'})
    dataset.to_csv(dataset_file, encoding='gb18030', index=False)


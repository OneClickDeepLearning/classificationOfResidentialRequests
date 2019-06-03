import csv
import pandas as pd
import time

file_path = "/Users/sunjincheng/Documents/nlpdata/company.csv"
valid = '/Users/sunjincheng/Documents/valid_data_all.csv'
invalid = '/Users/sunjincheng/Documents/nonvalid_all.csv'

def raw2valid(in_file,out_valid,out_inval):

    time1 = time.clock()
    csv_data = pd.read_csv(in_file, encoding="gb18030")  # 读取训练数据
    csv_data.sort_values(by='处置单位')
    csv_data.insert(12, '地点', None)
    time2 = time.clock()
    print(csv_data.shape)
    print("加载csv耗时：" + str(time2 - time1))
    valid_data = []
    null_data = []
    nonvalid_data = []
    dep_type = []
    department = []
    print("数据行数：" + str(csv_data.shape[0]))

    for i in range(0, csv_data.shape[0]):
        line = (csv_data.loc[i])
        if (pd.isnull(line['处置单位'])):
            nonvalid_data.append(line)
        elif (line['处置单位'] == "其他单位" or line['处置单位'] == "省外单位" or line['处置单位'] == "省级单位" or line['处置单位'] == "除海口外的市县" or
              line['处置单位'] == '无效归属' or line['处置单位'] == '无效数据' or line['处置单位'] == "政府单位"):
            nonvalid_data.append(line)
        else:
            valid_data.append(line)

            dep_type.append(line['处置单位'])
    time3 = time.clock()
    valid_num = len(valid_data)
    print("数据处理时间：" + str(time3 - time2))
    # null_num=len(null_data)

    print("有效数据：" + str(valid_num))
    print('无效数据：' + str(len(nonvalid_data)))

    valid_df = pd.DataFrame(valid_data)
    nonvalid_df = pd.DataFrame(nonvalid_data)
    # 排列

    valid_df = valid_df.sort_values(by='处置单位')
    nonvalid_df = nonvalid_df.sort_values(by='处置单位')

    valid_df.to_csv(out_valid, encoding="gb18030", index = False)
    nonvalid_df.to_csv(out_inval, encoding="gb18030", index = False)

    dep_num = set(dep_type)

    print("部门数量：" + str(len(dep_num)))
    time4 = time.clock()
    for line in dep_num:
        department.append([line, dep_type.count(line)])
    department.sort(key=lambda x: x[1])
    for i in range(0, len(department)):
        print(department[i][0] + str(department[i][1]))

def clean(valid_file):
    clean_data = []
    i = 0
    for line in csv.reader(open(valid_file,encoding='gb18030')):
        i += 1
        if i<=1:
            continue

        if(line[14] == '无效归属'):
            print(line[14])

# clean(valid)
#
# raw2valid(file_path,valid,invalid)
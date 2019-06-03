import csv
import re
import jieba
import pandas as pd

file1 = '/Users/sunjincheng/Desktop/Prof-project/NLPproject/data/valid_data_all.csv'
# df = pd.read_csv(file1,encoding='gb18030')
file2 = '../data/split_data.txt'
read_file = open(file1, encoding='gb18030')
lines = csv.reader(read_file)
count=0
lengths = []
write_file = open(file2,'a',encoding='utf-8')
for line in lines:
    count +=1
    if(count == 1):
        continue
    sent = line[8]
    sent = re.sub('市民来电咨询', '', sent)
    sent = re.sub('市民来电反映', '', sent)
    sent = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[a-zA-Z0-9+——！，。？、~@#￥%……&*（）《》：:]+", "",sent)
    splits = jieba.cut(sent)
    length = len(list(splits))
    lengths.append(length)
    # result = ' '.join(splits)
    # print(sent)
    # print(result)
    # print(result.split())

    # write_file.write(result)
    # write_file.write('\n')
    if (count % 10000 == 0):
        print(count)

write_file.close()
read_file.close()




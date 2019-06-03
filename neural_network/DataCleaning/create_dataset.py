from DataCleaning import raw2valid, valid2dataset, split_data
from Clustering.Optics_cluster import create_label
import pandas as pd

def raw2dataset():
    pass


valid_data = '/Users/sunjincheng/Documents/nlpdata/valid_data_all.csv'
all_label = '../data/all_labels.txt'
label = '../data/labels.txt'
dataset_file = '../data/dataset_all.csv'
train_set = '../data/trainset.csv'
test_set = '../data/testset.csv'


# Each time create dataset from valid data
# will rerun clustering, thus labels change
def valid2set(in_file, per):

    # lines, count = count_dept.count_dept(in_file, all_label)
    # #create_label(all_label, count, label)
    valid2dataset.valid2set(label, in_file, dataset_file)
    # splitData.split(dataset_file, int(lines * per), train_set, test_set)

def extract_data(dataset_file,num,train = True):
    if train:
        out_file = '../data/%d_trainset.csv'%(num)
    else:
        out_file = '../data/%d_testset.csv'%(num)
    # splitData.extract(dataset_file,num,out_file)

# valid2set(valid_data,0.8)
# extract_data(train_set,80000,True)
# extract_data(test_set,4000,False)

a = pd.read_csv(valid_data,encoding='gb18030')

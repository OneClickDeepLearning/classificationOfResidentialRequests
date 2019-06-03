import pandas as pd
from sklearn.utils import shuffle


def split(dataset_file, num, train_file, test_file):
    dataset = pd.read_csv(dataset_file, encoding='gb18030')
    dataset = shuffle(dataset)

    # def split(dataset,num):
    train_set = dataset[0:num]
    test_set = dataset[num:]
    train_set.to_csv(train_file, encoding='gb18030', index=False)
    test_set.to_csv(test_file, encoding='gb18030', index=False)


def extract(dataset_file, num, out_file):
    dataset = pd.read_csv(dataset_file, encoding='gb18030')
    dataset = shuffle(dataset)
    dataset = dataset[0:num]
    dataset.to_csv(out_file, encoding='gb18030', index=False)

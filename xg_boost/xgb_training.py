import csv
import pickle

import numpy as np
import xgboost as xgb

from tfidf import tfidf

DEBUG = True
TRAINING_DATA_FILE = '.csv'


def train_by_xgb(train_X_array, train_Y_array, test_X_array, test_Y_array, model_name, version):
    xg_train = xgb.DMatrix(train_X_array, label=train_Y_array)
    xg_test = xgb.DMatrix(test_X_array, label=test_Y_array)
    param = {}
    # usZe softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.07
    param['max_depth'] = 150
    param['silent'] = 1
    param['num_class'] = 53
    param['min_child_weight'] = 1
    param['gamma'] = 0.1
    param['nthread'] = 4

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 200
    model = xgb.train(param, xg_train, num_round, watchlist, xgb_model='{}{}.model'.format(model_name, version - 1),
                      early_stopping_rounds=30)
    model.save_model('{}{}.model'.format(model_name, version))

    pred = model.predict(xg_test)
    print('predicting, classification error=%f' % (
            sum(int(pred[i]) != test_Y_array[i] for i in range(len(test_Y_array))) / float(len(test_Y_array))))


def run_xgb(test_data, test_label):
    test_X_array = np.array(test_data, dtype=float, ndmin=2)
    test_Y_array = np.array(test_label, dtype=int).reshape(len(test_label), 1)

    training_label = []
    training_data = []
    # generate training data
    with open(TRAINING_DATA_FILE, 'r', encoding='GB18030') as db01:
        reader = csv.reader(db01)
        for i, row in enumerate(reader):
            # get responsible department tfidf
            temp = tfidf.get_instance_tfidf_vector(str(row[0]), True)
            training_data.append(temp[:-1])
            training_label.append(int(row[9]))

    train_X_array = np.array(training_data, dtype=float, ndmin=2)
    train_Y_array = np.array(training_label, dtype=int).reshape(len(training_data), 1)

    train_by_xgb(train_X_array, train_Y_array, test_X_array, test_Y_array, 'model_v1', 1)


def read_pickle(file_path):
    data = ""
    if not file_path.endswith(".pickle"):
        print("[ERROR] file suffix missing or file suffix is not appropriate.")
    else:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    return data


def main(filename):
    test_data_input = read_pickle(f'{filename}')
    test_data = [i[-1] for i in test_data_input]
    test_label = [i[:-1] for i in test_data_input]
    run_xgb(test_data, test_label)


if __name__ == '__main__':
    main(filename='')

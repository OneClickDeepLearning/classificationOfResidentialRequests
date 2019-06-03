import pandas as pd
import numpy as np
from keras import models, Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, Activation, LeakyReLU, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils
from keras import regularizers
from keras import optimizers
from keras.callbacks import EarlyStopping


# def train_model(classes):
classes = 5
num = 40000
X_train = np.load('../data/train_x.npy')[0:num]
Y_train = np.load('../data/train_y.npy')[0:num]

# X_train reshape to [40000,100,100]
X_train = np.reshape(X_train, [40000, 100, 100, 1])
# Y_train encode to one-hot format
encoder = preprocessing.LabelEncoder()
encoder.fit_transform(Y_train)
Y_train = encoder.transform(Y_train)
Y_train = np_utils.to_categorical(Y_train, num_classes=classes)
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

def train(data_x, data_y, val_x, val_y, name):
    model = Sequential()
    model.add(Conv2D(filters=256,
                     input_shape=[100, 100, 1],
                     kernel_size=[3, 100],
                     kernel_initializer='random_uniform',
                     kernel_regularizer=regularizers.l2(0.01),
                     strides=[1, 100],
                     padding='same',
                     # activity_regularizer=regularizers.l1(0.1)
                     ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=256,
                    kernel_initializer='random_uniform',
                    kernel_regularizer=regularizers.l2(0.01),
                    # activity_regularizer=regularizers.l1(0.01)
                    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(units=64,
                    kernel_initializer='random_uniform',
                    kernel_regularizer=regularizers.l2(0.01),
                    # activity_regularizer=regularizers.l1(0.01)
                    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    #
    model.add(Dense(units=32,
                    kernel_initializer='random_uniform',
                    kernel_regularizer=regularizers.l2(0.01),
                    # activity_regularizer=regularizers.l1(0.01)
                    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(units=5,
                    kernel_initializer='random_uniform',
                    kernel_regularizer=regularizers.l2(0.01),
                    # activity_regularizer=regularizers.l1(0.1)
                    ))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    #
    # model.add(Dense(units=classes,
    #                 activation='softmax',
    #                 activity_regularizer=regularizers.l1(0.01),
    #                 kernel_initializer='random_uniform',
    #                 kernel_regularizer=regularizers.l2(0.01),
    #                 bias_initializer='zeros'))
    model.summary()
    adam = optimizers.adam(lr=0.005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    # model = models.load_model('0.753CNN')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

    model.fit(data_x, data_y,
              batch_size=500,
              epochs=100,
              validation_data=(val_x, val_y),
              callbacks=[monitor])
    score = model.evaluate(val_x, val_y, verbose=0)
    val_loss = score[0]
    acc = score[1]
    model.save(name + "kCNNmodel")
    X_test = np.load('../data/test_x.npy')
    Y_test = np.load('../data/test_y.npy')
    # X_train reshape to [40000,100,100]
    X_test = np.reshape(X_test, [4000, 100, 100, 1])
    score = accuracy_score(model.predict_classes(X_test), Y_test)
    print(score)
    return val_loss, acc


def k_ford():
    num = 40000
    X_train = np.load('../data/train_x.npy')[0:num]
    Y_train = np.load('../data/train_y.npy')[0:num]
    scores = []
    count = 0
    LOSS = 100
    X_train = np.reshape(X_train, [num, 100, 100, 1])
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
    for trainset, testset in kfold.split(X_train, Y_train):
        encoder = preprocessing.LabelEncoder()
        Y_train_new = encoder.fit_transform(Y_train)
        # Y_train_new = encoder.transform(Y_train)
        Y_train_new = np_utils.to_categorical(Y_train_new, num_classes=5)
        loss, score = train(X_train[trainset], Y_train_new[trainset], X_train[testset], Y_train_new[testset],
                            str(count))
        if (loss < LOSS):
            scores.append(score)
            LOSS = loss
            count += 1
        else:
            print(scores)
            break


def train2(data_x, data_y, val_x, val_y, name):
    model = Sequential()
    model.add(Conv2D(filters=256,
                     input_shape=[100, 100, 1],
                     kernel_size=[3, 100],
                     kernel_initializer='random_uniform',
                     kernel_regularizer=regularizers.l2(0.01),
                     strides=[1, 100],
                     padding='same',
                     # activity_regularizer=regularizers.l1(0.1)
                     ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=256,
                    kernel_initializer='random_uniform',
                    kernel_regularizer=regularizers.l2(0.01),
                    # activity_regularizer=regularizers.l1(0.01)
                    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(units=64,
                    kernel_initializer='random_uniform',
                    kernel_regularizer=regularizers.l2(0.01),
                    # activity_regularizer=regularizers.l1(0.01)
                    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    #
    model.add(Dense(units=32,
                    kernel_initializer='random_uniform',
                    kernel_regularizer=regularizers.l2(0.01),
                    # activity_regularizer=regularizers.l1(0.01)
                    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(units=5,
                    kernel_initializer='random_uniform',
                    kernel_regularizer=regularizers.l2(0.01),
                    # activity_regularizer=regularizers.l1(0.1)
                    ))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    #
    # model.add(Dense(units=classes,
    #                 activation='softmax',
    #                 activity_regularizer=regularizers.l1(0.01),
    #                 kernel_initializer='random_uniform',
    #                 kernel_regularizer=regularizers.l2(0.01),
    #                 bias_initializer='zeros'))
    model.summary()
    adam = optimizers.adam(lr=0.005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

    model.fit(data_x, data_y,
              batch_size=500,
              epochs=20,
              validation_data=(val_x, val_y),
              callbacks=[monitor])
    score = model.evaluate(val_x, val_y, verbose=0)
    val_loss = score[0]
    acc = score[1]
    model.save(name + "../models/kCNNmodel")
    X_test = np.load('../data/test_x.npy')
    Y_test = np.load('../data/test_y.npy')
    # X_train reshape to [40000,100,100]
    X_test = np.reshape(X_test, [4000, 100, 100, 1])
    score = accuracy_score(model.predict_classes(X_test), Y_test)

    print(score)
    return val_loss, acc

def test(model):
    model = models.load_model(model)

    X_test = np.load('../data/test_x.npy')
    Y_test = np.load('../data/test_y.npy')

    # X_train reshape to [40000,100,100]
    X_test = np.reshape(X_test, [4000, 100, 100, 1])
    score = accuracy_score(model.predict_classes(X_test), Y_test)

    encoder = preprocessing.LabelEncoder()
    Y_train_new = encoder.fit_transform(Y_test)
    # Y_train_new = encoder.transform(Y_train)
    Y_train_new = np_utils.to_categorical(Y_train_new, num_classes=5)
    loss = model.evaluate(X_test, Y_train_new)[0]
    print(loss)


# train_model(162)
def catchOn(model):
    model = models.load_model(model)
    model.fit(x_train, y_train,
              batch_size=500,
              epochs=10,
              validation_data=(x_val, y_val))


k_ford()

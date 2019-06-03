import pandas as pd
import numpy as np
from keras import models,Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils
from keras.callbacks import EarlyStopping




def train_model():
    num = 80000
    X_train = np.load('../data/train_x.npy')[0:num]
    Y_train = np.load('../data/train_y.npy')[0:num]
    # Y_train = Y_train[:,1]
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    y_train = pd.DataFrame(y_train)[0]
    y_val = pd.DataFrame(y_val)[0]
    # one-hot，5 category
    y_labels = list(y_train.value_counts().index)
    y_labels = list(range(157))
    # y_labels = np.unique(y_train)
    le = preprocessing.LabelEncoder()
    le.fit(y_labels)
    num_labels = len(y_labels)
    y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), 157)
    y_val = to_categorical(y_val.map(lambda x: le.transform([x])[0]), 157)
    model = Sequential()
    model.add(Dense(1024, input_shape=(x_train.shape[1],), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(157, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # model = models.load_model('0.7305NN')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    model.fit(x_train, y_train,
              batch_size=500,
              epochs=50,
              validation_data=(x_val, y_val),
              callbacks=[monitor])
    model.save("../models/80000NN.h5py")
    X_test = np.load('../data/test_x.npy')
    Y_test = np.load('../data/test_y.npy')
    Y_test = Y_test[:,1]
    score = accuracy_score(model.predict_classes(X_test), Y_test)

    print(score)

def k_fold():
    num = 40000
    X_train = np.load('../data/train_x.npy')[0:num]
    Y_train = np.load('../data/train_y.npy')[0:num]
    scores = []
    #one-hot
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
    for train, test in kfold.split(X_train, Y_train):
        encoder = preprocessing.LabelEncoder()
        encoder.fit_transform(Y_train)
        Y_train_new = encoder.transform(Y_train)
        Y_train_new = np_utils.to_categorical(Y_train_new, num_classes=5)
        model = Sequential()
        model.add(Dense(1024, input_shape=(X_train.shape[1],), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(5, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(X_train[train], Y_train_new[train],
                  batch_size=500,
                  epochs=10)
        score=model.evaluate(X_train[test],Y_train_new[test],verbose=0)
        scores.append(score[1])
    return scores



# X_test = np.load('../data/test_x.npy')
# Y_test = np.load('../data/test_y.npy')
# model=models.load_model("0.7305NN")
# score = accuracy_score(model.predict_classes(X_test), Y_test)
# print(score)


train_model()
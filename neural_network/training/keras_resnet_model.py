import pandas as pd
import numpy as np
from keras import models, Sequential
from keras.layers import BatchNormalization, Conv2D, Activation, Dropout, GlobalAveragePooling2D, add, Input, Dense,MaxPool2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils
from keras import regularizers
from keras import optimizers
from keras.callbacks import EarlyStopping
import sys
sys.path.append('/home/oneclick/CFSC/CFSC/')
from training.resnet import Res50

def bn_relu(layer, dropout=0, **params):
    layer = BatchNormalization()(layer)
    layer = Activation(params['conv_activation'])(layer)

    if dropout > 0:
        layer = Dropout(dropout)(layer)
    return layer


def resnet_block_A(layer, filters, kernels, dropout, activation, cross_block=False,shrink = False):
    # -Conv-BN-Act-Conv-BN-Act-
    # ↳-----Conv-BN-Act-------↑

    shape = [1, 1]
    if shrink:
        shape = [2, 2]

    if cross_block:
        shortcut = Conv2D(filters=filters,
                          kernel_size=shape,
                          kernel_initializer='random_uniform',
                          # kernel_regularizer=regularizers.l2(0.01),
                          strides=shape,
                          padding='same')(layer)
        shortcut = bn_relu(shortcut,conv_activation='relu')
    else:
        shortcut = layer

    layer = Conv2D(filters=filters,
                   kernel_size=kernels,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=shape,
                   padding='same')(layer)
    layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = Conv2D(filters=filters,
                   kernel_size=kernels,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=[1, 1],
                   padding='same')(layer)


    layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = add([shortcut, layer])

    # if shrink:
    #     layer = MaxPool2D()(layer)

    return layer


def resnet_block_B(layer, filters, kernels, dropout, activation,
                   cross_block=False, is_first=False, is_last=False, shrink = False):
    # -BN-Act-Conv-BN-Act-Conv--
    # ↳-----------------------↑
    strides = [1, 1]
    if shrink:
        strides = [2, 2]

    if cross_block:

        shortcut = Conv2D(filters=filters,
                          kernel_size=strides,
                          kernel_initializer='random_uniform',
                          # kernel_regularizer=regularizers.l2(0.01),
                          strides=strides,
                          padding='same')(layer)
    else:
        shortcut = layer

    if not is_first:
        layer = bn_relu(layer, dropout=dropout, conv_activation=activation)



    layer = Conv2D(filters=filters,
                   kernel_size=kernels,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=strides,
                   padding='same')(layer)
    layer = bn_relu(layer, dropout=dropout, conv_activation=activation)


    layer = Conv2D(filters=filters,
                   kernel_size=kernels,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=[1, 1],
                   padding='same')(layer)
    layer = add([shortcut,layer])

    if is_last:
        layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    return layer


def global_average_pooling(layer, cls):
    layer = Conv2D(cls, [1, 1])(layer)
    layer = GlobalAveragePooling2D()(layer)
    layer = Activation(activation='softmax')(layer)
    return layer


def Resnet_Comparation(num):
    input = Input(shape=[100, 100, 1])
    #100*100

    layer = resnet_block_B(input, 32, [3, 3], 0, 'relu', is_first=True)
    layer = resnet_block_B(layer, 32, [3, 3], 0, 'relu')
    #50*50
    layer = resnet_block_B(layer, 64, [3, 3], 0, 'relu', cross_block=True, shrink=True)
    layer = resnet_block_B(layer, 64, [3, 3], 0, 'relu')
    #25*25
    layer = resnet_block_B(layer, 128, [3, 3], 0, 'relu', cross_block=True, shrink=True)
    layer = resnet_block_B(layer, 128, [3, 3], 0, 'relu')
    #13*13
    layer = resnet_block_B(layer, 256, [3, 3], 0, 'relu', cross_block=True, shrink=True)
    layer = resnet_block_B(layer, 256, [3, 3], 0, 'relu')
    #7*7
    layer = resnet_block_B(layer, 512, [3, 3], 0, 'relu', cross_block=True, shrink=True)
    layer = resnet_block_B(layer, 512, [3, 3], 0, 'relu', is_last=True)

    output = global_average_pooling(layer, num)
    model = Model(inputs = [input], outputs = [output])
    model.summary()
    return model

def Resnet_A(num):
    input = Input(shape=[100, 100, 1])
    #100*100
    layer1 = resnet_block_A(layer = input, filters= 32, kernels=[3,3],
                           dropout= 0,activation = 'relu',cross_block= True)
    layer2 = resnet_block_A(layer = layer1, filters= 32, kernels=[3,3],
                           dropout= 0,activation = 'relu')
    #50*50
    layer3 = resnet_block_A(layer=layer2, filters=64, kernels=[3, 3],
                           dropout=0, activation='relu',cross_block= True,shrink=True)
    layer4 = resnet_block_A(layer = layer3, filters= 64, kernels=[3,3],
                           dropout= 0,activation = 'relu')
    #25*25
    layer5 = resnet_block_A(layer=layer4, filters=128, kernels=[3, 3],
                           dropout=0, activation='relu', cross_block=True, shrink=True)
    layer6 = resnet_block_A(layer=layer5, filters=128, kernels=[3, 3],
                                   dropout=0, activation='relu')
    #13*13
    layer7 = resnet_block_A(layer=layer6, filters=256, kernels=[3, 3],
                           dropout=0, activation='relu', cross_block=True, shrink=True)
    layer8 = resnet_block_A(layer=layer7, filters=256, kernels=[3, 3],
                                   dropout=0, activation='relu')
    #7*7
    layer9 = resnet_block_A(layer=layer8, filters=512, kernels=[3, 3],
                           dropout=0, activation='relu', cross_block=True, shrink=True)
    layer10 = resnet_block_A(layer=layer9, filters=512, kernels=[3, 3],
                                   dropout=0, activation='relu')


    output = global_average_pooling(layer10, num)
    model = Model(inputs = [input], outputs = [output])
    model.summary()
    return model
def ResnetB(dense = False):
    input = Input(shape=[100, 100, 1])
    layer = Conv2D(filters=32,
                   kernel_size=[3, 3],
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=[1, 1],
                   padding='same')(input)
    layer = resnet_block_B(layer, 32, [3, 3], 0, 'relu')
    # 50*50
    layer = resnet_block_B(layer, 64, [3, 3], 0, 'relu', cross_block=True, shrink=True)
    layer = resnet_block_B(layer, 64, [3, 3], 0, 'relu')
    # 25*25
    layer = resnet_block_B(layer, 128, [3, 3], 0, 'relu', cross_block=True, shrink=True)
    layer = resnet_block_B(layer, 128, [3, 3], 0, 'relu')
    # 13*13
    layer = resnet_block_B(layer, 256, [3, 3], 0, 'relu', cross_block=True, shrink=True)
    layer = resnet_block_B(layer, 256, [3, 3], 0, 'relu')
    # 7*7
    layer = resnet_block_B(layer, 512, [3, 3], 0, 'relu', cross_block=True, shrink=True)
    layer = resnet_block_B(layer, 512, [3, 3], 0, 'relu', is_last=True)
    if dense:
        output = Dense(units=5,activation='softmax')(layer)
    else:
        output = global_average_pooling(layer, 5)
    model = Model(inputs=[input], outputs=[output])
    model.summary()
    return model

def train(model):
    num = 80000
    X_train = np.load('../data/train_x.npy')[0:num]
    Y_train = np.load('../data/train_y.npy')[0:num]
    #Y_train = Y_train[:]
    X_train = np.reshape(X_train, [num, 100, 100, 1])
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
    y_train = pd.DataFrame(y_train)[0]
    y_val = pd.DataFrame(y_val)[0]
    # one-hot，5 category
    y_labels = list(y_train.value_counts().index)
    y_labels = list(range(157))
    # y_labels = np.unique(y_train)
    le = preprocessing.LabelEncoder()
    le.fit(y_labels)
    num_labels = len(y_labels)
    y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)
    y_val = to_categorical(y_val.map(lambda x: le.transform([x])[0]), num_labels)

    adam = optimizers.adam(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')

    model.fit(x_train, y_train,
              batch_size=50,
              epochs=20,
              validation_data=(x_val, y_val),
              callbacks=[monitor])
    score = model.evaluate(x_val, y_val, verbose=0)
    val_loss = score[0]
    acc = score[1]
    model.save('../models/RCNN')
    X_test = np.load('../data/test_x.npy')
    Y_test = np.load('../data/test_y.npy')
    # X_train reshape to [40000,100,100]
    X_test = np.reshape(X_test, [4000, 100, 100, 1])
    Y_test = Y_test[:,0]
    score = model.predict(X_test)
    score = np.argmax(score, axis=1)
    score = accuracy_score(score, Y_test)

    print(score)

def test():
    model = models.load_model('../models/RCNN')
    X_test = np.load('../data/test_x.npy')
    Y_test = np.load('../data/test_y.npy')
    # X_train reshape to [40000,100,100]
    X_test = np.reshape(X_test, [4000, 100, 100, 1])
    Y_test = Y_test[:,0]
    score = model.predict(X_test)
    score = np.argmax(score, axis=1)
    print(score)
    print(Y_test)
    score = accuracy_score(score, Y_test)
    print(score)


#
# test()
#model = Resnet_A(157)
#from Adaptors.Networks import Res50
model = ResnetB(True)
train(model)

def conti_train():
    model = models.load_model('RCNN2')
    train(model)
# conti_train()
# model = ResnetB()
# train(model)

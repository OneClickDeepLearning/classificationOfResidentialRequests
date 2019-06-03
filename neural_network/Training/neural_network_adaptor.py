from Training.train_adaptor import Training_adaptor
import numpy as np


class NN_training_adaptor(Training_adaptor):

    def __init__(self):
        super().__init__()

    def load_data(self, x_train, y_train, x_test, y_test, **kwargs):

        self.X_train = np.load(x_train)
        self.Y_train = np.load(y_train)
        self.X_test = np.load(x_test)
        self.Y_test = np.load(y_test)
        self.Data_dim = self.X_train.shape[1]
        print('[INFO] Finish loading data')

    def label_data(self, **kwargs):
        # X_data: array[data_num, data_dim]                 exp:[40000,10000]
        # Y_data: array[data_num, layer_num]                exp:[40000,2]
        # layer: int, which layer of model
        # expand: int, which label to expand
        # label_num: how many labels if expanded, defult: automatically calculated
        import pandas as pd
        x_data = kwargs['X_data']
        y_data = kwargs['Y_data']
        layer = kwargs['layer']
        try:
            expand = kwargs['expand']
        except:
            pass
        Y_label = []
        if layer == 0:
            Y_label = y_data[:, 0]
            X_train = x_data
            Y_train = Y_label
            num_labels = Y_label.max() + 1
        else:
            for line in y_data:
                Y_label.append(line[layer] if line[layer - 1] == expand else -1)
            Y_label = np.array(Y_label)
            X_train = x_data
            Y_train = Y_label
            try:
                num_labels = kwargs['labels_num']
            except:
                num_labels = Y_label.max() + 1
            # ---------------
            X_train_new = []
            Y_train_new = []
            for x, y in zip(X_train, Y_train):
                if y != -1:
                    X_train_new.append(x)
                    Y_train_new.append(y)
            X_train = np.array(X_train_new)
            Y_train = np.array(Y_train_new)

            # ---------------------------------
        Y_train = pd.DataFrame(Y_train)[0]

        print('[INFO] Finish labeling data')
        return X_train, Y_train, num_labels

    def create_dataset(self, **kwargs):
        # X_train: array[data_num, data_dim]                exp: [12354,10000]
        # Y_train: array[data_num, 1]                       exp: [12354,1]
        # num_labels
        from sklearn.model_selection import train_test_split
        from sklearn import preprocessing
        from keras.utils.np_utils import to_categorical
        X_train = kwargs['X_train']
        Y_train = kwargs['Y_train']
        num_labels = kwargs['num_labels']
        x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

        y_labels = list(range(num_labels))
        le = preprocessing.LabelEncoder()
        le.fit(y_labels)
        y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)
        y_val = to_categorical(y_val.map(lambda x: le.transform([x])[0]), num_labels)
        print('[INFO] Dataset created and divided')
        return x_train, y_train, x_val, y_val

    def network(self, **kwargs):

        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        num = kwargs['cls_num']
        model = Sequential()
        model.add(Dense(512, input_shape=(self.Data_dim,), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num, activation='softmax'))
        model.summary()
        self.model = model

    def train_model(self, X_train, Y_train, **kwargs):
        from keras.callbacks import EarlyStopping

        x_val = kwargs['x_val']
        y_val = kwargs['y_val']
        model_path = kwargs['save_path']
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        # model = models.load_model('0.7305NN')
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto',)
        self.model.fit(X_train, Y_train,
                       batch_size=500,
                       epochs=50,
                       validation_data=(x_val, y_val),
                       callbacks=[monitor])
        self.model.save(model_path)
        del self.model

    def load_model(self, model_path):
        from keras.models import load_model
        self.model = load_model(model_path)

    def assesment(self, X_test, Y_test):
        from sklearn.metrics import accuracy_score
        y_pred = self.model.predict_classes(X_test)
        score = accuracy_score(y_pred, Y_test)
        print(score)

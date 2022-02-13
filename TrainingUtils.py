import numpy as np

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPool2D

class TrainingUtils:
    @staticmethod
    def train(model, x, y, epochs, modelname='model'):
        model.fit(x, y, epochs = epochs)
        model.save(modelname + '.h5')

    @staticmethod
    def prepare_input(x_raw, y_raw):
        x = np.array(x_raw)
        y = to_categorical(y_raw).astype(int)
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.05)
        x_train, x_eval, y_train, y_eval = train_test_split(x_train,y_train, test_size = 0.05)
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        x_eval = np.expand_dims(x_eval, axis=3)
        inputs = {
            'x_train' : x_train,
            'x_test' : x_test,
            'x_eval' : x_eval,
            'y_train' : y_train,
            'y_test' : y_test,
            'y_eval' : y_eval
        }
        return inputs

    @staticmethod
    def createmodel():
        model = Sequential()
        model.add(Conv2D(32, (3,3),
                        activation='relu',
                        input_shape=(258,13,1)))
        model.add(MaxPool2D((3, 3), strides=(2,2,), padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3,3),
                        activation='relu',
                        input_shape=(258,13,1)))
        model.add(MaxPool2D((3, 3), strides=(2,2,), padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (2,2),
                        activation='relu',
                        input_shape=(258,13,1)))
        model.add(MaxPool2D((2, 2), strides=(2,2,), padding='same'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(64,activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model

    
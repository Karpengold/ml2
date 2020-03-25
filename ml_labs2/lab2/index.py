import os
import cv2
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import imagehash
import time

TRAIN_IMAGES_PATH = 'C:\\Users\karps\Downloads\\notMNIST_large\\notMNIST_large'
TEST_IMAGES_PATH = 'C:\\Users\\karps\\PycharmProjects\\untitled\\ml_labs2\lab1\\notMNIST_small'

def show_learning_process():
    # Plot history: MAE
    plt.plot(history.history['loss'], label='MAE (testing data)')
    plt.plot(history.history['val_loss'], label='MAE (validation data)')
    plt.title('MAE for Chennai Reservoir Levels')
    plt.ylabel('MAE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()

def get_class(letter):
    return ord(letter) - ord('A')

def load_data(IMAGES_PATH, minimum=36806):
    X = []
    y = []
    start = time.time()
    dirs = os.listdir(IMAGES_PATH)
    print(minimum)
    for dir in dirs:
        print(dir)
        images = os.listdir(os.path.join(IMAGES_PATH, dir))
        i=0
        while i < minimum:
            path = os.path.join(IMAGES_PATH, dir, images[i])
            try:
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE).flatten()
            except:
                print('Bad image')
                i += 1
                continue
            if image is not None:
                X.append(image)
                y.append(get_class(dir))
            i +=1
    print('Dataset length: ', len(X))
    print('Time: ', time.time() - start)
    return X, y

from keras.models import Sequential

model = Sequential()


X_test, y_test = load_data(TEST_IMAGES_PATH, 1498)
X, y = load_data(TRAIN_IMAGES_PATH, 2000)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras import regularizers
import numpy as np
from keras.utils import np_utils


#1-2
model = Sequential()
model.add(Dense(units=300, activation='relu', input_dim=784))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=30, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X, np_utils.to_categorical(y), batch_size=32, epochs=50)
y_pred = np.argmax(model.predict(np.array(X_test)), axis=1)

print(accuracy_score(y_pred, y_test))


#3
model = Sequential()
model.add(Dense(units=300, activation='relu', input_dim=784, kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(units=100, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(units=50, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(units=30, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X, np_utils.to_categorical(y), batch_size=32, epochs=50)
y_pred = np.argmax(model.predict(np.array(X_test)), axis=1)

print(accuracy_score(y_pred, y_test))



#4
model = Sequential()
epochs = 50
opt = SGD(lr=1e-2, momentum=0.9, decay=1e-2/epochs)
model.add(Dense(units=300, activation='relu', input_dim=784, kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(units=100, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(units=50, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(units=30, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history = model.fit(X, np_utils.to_categorical(y), batch_size=32, epochs=epochs)

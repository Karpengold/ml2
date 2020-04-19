import os
import cv2
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import imagehash
import time
from PIL import Image
TRAIN_IMAGES_PATH = 'C:\\Users\karps\Downloads\\notMNIST_large\\notMNIST_large'
TEST_IMAGES_PATH = 'C:\\Users\\karps\\PycharmProjects\\untitled\\ml_labs2\lab1\\notMNIST_small'

def get_class(letter):
    return ord(letter) - ord('A')

def remove_duplicates(TRAIN_IMAGE_PATH, TEST_IMAGE_PATH):
    train_dirs = os.listdir(TRAIN_IMAGE_PATH)
    test_dirs = os.listdir(TEST_IMAGE_PATH)

    start = time.time()
    hashes = set()
    i = 0
    copies = 0
    for dir in test_dirs:
        print(dir)
        images = os.listdir(os.path.join(TEST_IMAGE_PATH, dir))

        for img in images:
            path = os.path.join(TEST_IMAGE_PATH, dir, img)
            try:
                image = Image.open(path)
            except:
                os.remove(path)
                continue
            hash = imagehash.phash(image)
            if hash in hashes:
                copies += 1
                os.remove(path)
            else:
                hashes.add(hash)

    for dir in train_dirs:
        print(dir)
        images = os.listdir(os.path.join(TRAIN_IMAGE_PATH, dir))

        for img in images:
            path = os.path.join(TRAIN_IMAGE_PATH, dir, img)
            try:
                image = Image.open(path)
            except:
                os.remove(path)
                continue
            hash = imagehash.phash(image)
            if hash in hashes:
                copies += 1
                os.remove(path)
            else:
                hashes.add(hash)

    i += 1
    print('Time:', time.time() - start)
    print('Count of duplciates:', copies)


def load_data(IMAGES_PATH, minimum=36298, minimum_test=1400):
    X = []
    y = []
    X_test = []
    y_test = []
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
                image = (cv2.imread(path, cv2.IMREAD_GRAYSCALE).flatten())/255
            except:
                print('Bad image')
                i += 1
                continue
            if image is not None:
                X.append(image)
                y.append(get_class(dir))
            i +=1
        while i < minimum + minimum_test:
            path = os.path.join(IMAGES_PATH, dir, images[i])
            try:
                image = (cv2.imread(path, cv2.IMREAD_GRAYSCALE).flatten()) / 255
            except:
                print('Bad image')
                i += 1
                continue
            if image is not None:
                X_test.append(image)
                y_test.append(get_class(dir))
            i += 1
    print('Train dataset length: ', len(X))
    print('Test dataset length: ', len(X_test))
    print('Time: ', time.time() - start)
    random_order = np.random.permutation(len(X))
    X = np.array(X)
    y = np.array(y)
    return X[random_order], y[random_order],np.array(X_test), np.array(y_test)

def get_balance_value(IMAGES_PATH):
    minimum = math.inf
    dirs = os.listdir(IMAGES_PATH)
    for dir in dirs:
        images = os.listdir(os.path.join(IMAGES_PATH, dir))
        if len(images) < minimum:
            minimum = len(images)
    return minimum



#remove_duplicates(TRAIN_IMAGES_PATH, TEST_IMAGES_PATH)
#remove_duplicates(TEST_IMAGES_PATH)
print(get_balance_value(TEST_IMAGES_PATH))
print(get_balance_value(TRAIN_IMAGES_PATH))


train_sizes = [5, 10, 100, 500, 1000, 2000, 5000, 10000, 20000]
X, y, X_test, y_test = load_data(TRAIN_IMAGES_PATH, 20000)
accuracy_scores = []
for size in train_sizes:
    X, y, _, __ = load_data(TRAIN_IMAGES_PATH, size)
    print('Train size:', size)
    start = time.time()
    clf = LogisticRegression().fit(X, y)
    accuracy_scores.append(accuracy_score(clf.predict(X_test), y_test))
    print('Time:', time.time() - start)


plt.plot(train_sizes, accuracy_scores)
plt.xlabel('Train dataset size')
plt.ylabel('Accuracy')
plt.show()






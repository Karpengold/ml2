import os
import cv2
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import imagehash
import time
from PIL import Image
TRAIN_IMAGES_PATH = 'C:\\Users\karps\Downloads\\notMNIST_large\\notMNIST_large'
TEST_IMAGES_PATH = 'C:\\Users\\karps\\PycharmProjects\\untitled\\ml_labs2\lab1\\notMNIST_small'

def get_class(letter):
    return ord(letter) - ord('A')

def remove_duplicates(IMAGES_PATH):
    dirs = os.listdir(IMAGES_PATH)
    start = time.time()
    hashes = set()
    i = 0
    copies = 0
    for dir in dirs:
        print(dir)
        images = os.listdir(os.path.join(IMAGES_PATH, dir))

        for img in images:
            path = os.path.join(IMAGES_PATH, dir, img)
            try:
                image =Image.open(path)
            except:
                print('Bad image')
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

def get_balance_value(IMAGES_PATH):
    minimum = math.inf
    dirs = os.listdir(IMAGES_PATH)
    for dir in dirs:
        images = os.listdir(os.path.join(IMAGES_PATH, dir))
        if len(images) < minimum:
            minimum = len(images)
    return minimum



#remove_duplicates(TRAIN_IMAGES_PATH)
#remove_duplicates(TEST_IMAGES_PATH)
#print(get_balance_value(TEST_IMAGES_PATH))

X_test, y_test = load_data(TEST_IMAGES_PATH, 1498)
X, y = load_data(TRAIN_IMAGES_PATH, 36805)

train_sizes = [50, 100, 1000, 5000, 10000, 20000, 50000]
accuracy_scores = []
for size in train_sizes:
    print('Train size:', size)
    start = time.time()
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=size, test_size=10000)
    clf = LogisticRegression().fit(X_train, y_train)
    accuracy_scores.append(accuracy_score(clf.predict(X_test), y_test))
    print('Time:', time.time() - start)

plt.plot(train_sizes, accuracy_scores)
plt.xlabel('Train dataset size')
plt.ylabel('Accuracy')
plt.show()






import os
import glob

import numpy as np
from PIL import Image
from keras import regularizers
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.utils import np_utils
import matplotlib.pyplot as plt

root_dir = "14_kfood\\kfood"
categories = ['Chicken', 'Dolsotbab', 'Jeyugbokk-eum', 'Kimchi', 'Samgyeobsal', 'SoybeanPasteStew']

nb_classes = len(categories)  # 6

image_width = 64
image_height = 64

X = []  # 훈련데이터 이미지(검증데이터 포함)
Y = []  # 훈련데이터 레이블(검증데이터 포함)
X_t = []  # 테스트데이터 이미지
y_t = []  # 테스트데이터 레이블

# Train Dataset Load
for idx, category in enumerate(categories):
    image_dir = os.path.join(os.getcwd(), root_dir, category)
    files = glob.glob(os.path.join(image_dir, "*.jpg"))
    print(image_dir + "/" + "*.jpg")
    for f in files:
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_width, image_height))
        data = np.asarray(img)
        X.append(data)
        Y.append(idx)

# Test Dataset Load
for idx, category in enumerate(categories):
    image_dir = os.path.join(os.getcwd(), root_dir, "Testdata", category)
    files = glob.glob(os.path.join(image_dir, "*.jpg"))
    print(image_dir + "/" + "*.jpg")
    for f in files:
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_width, image_height))
        data = np.asarray(img)
        X_t.append(data)
        y_t.append(idx)

X = np.array(X)
Y = np.array(Y)
X_t = np.array(X_t)
y_t = np.array(y_t)


# 데이터 로딩
def load_datasets():
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
    # X_train = X_train.astype("float").reshape(-1, 64, 64, 3) / 255
    # X_val = X_val.astype("float").reshape(-1, 64, 64, 3) / 255
    # X_test = X_t.astype("float").reshape(-1, 64, 64, 3) / 255
    X_train = X_train.reshape(-1, 64, 64, 3) / 255
    X_val = X_val.reshape(-1, 64, 64, 3) / 255
    X_test = X_t.reshape(-1, 64, 64, 3) / 255
    y_train = np_utils.to_categorical(y_train, 6)
    y_val = np_utils.to_categorical(y_val, 6)
    y_test = np_utils.to_categorical(y_t, 6)
    return X_train, X_val, y_train, y_val, X_test, y_test


# 모델 구성
def build_model(in_shape):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, strides=3, padding="same", activation="relu", input_shape=in_shape))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=3, strides=3, padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=3, strides=3, padding="valid"))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=512), activation="relu")
    model.add(Dropout(0.5))
    model.add(Dense(units=6), activation="softmax")
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    return model


def model_train(x, y):
    model = build_model(x.shape[1:])
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=20, restore_best_weights=True)
    history = model.fit(x, y, batch_size=16, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping_cb])
    return model, history


def model_eval(model, x, y):
    score = model.evaluate(x, y)
    print("loss = ", score[0])
    print("accuracy = ", score[1])


def model_plot():
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.show()


if __name__ == '__main__':
    X_train, X_val, y_train, y_val, X_test, y_test = load_datasets()
    model, history = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)
    model_plot()

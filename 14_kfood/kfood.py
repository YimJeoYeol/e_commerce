# #%%
# import glob
#
# import numpy as np
# from PIL import Image
# from sklearn.model_selection import train_test_split
# #%%
# root_dir = "./kfood"
# categories = ['Chicken',
#               'Dolsotbab',
#               'Jeyugbokk-eum',
#               'Kimchi',
#               'Samgyeobsal',
#               'SoybeanPasteStew']
# #%%
# nb_classes = len(categories)  # 6
# #%%
# image_width = 64
# image_height = 64
# #%%
# X = []  # 이미지 데이터
# Y = []  # 레이블 데이터
# #%%
# for idx, category in enumerate(categories):
#     image_dir = root_dir + "/" + category
#     files = glob.glob(image_dir + "/" + "*.jpg")
#     print(image_dir + "/" + "*.jpg")
#     for f in files:
#         img = Image.open(f)
#         img = img.convert("RGB")
#         img = img.resize((image_width, image_height))
#         data = np.asarray(img)
#         X.append(data)
#         Y.append(idx)
#
# X = np.array(X)
# Y = np.array(Y)
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y)
# xy = (X_train, X_test, y_train, y_test)
# np.save(root_dir + "/kfood.npy", xy)
#%%
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.utils import np_utils

image_size = 64
#%%
# 데이터 로딩
def load_datasets():
    # X_train, X_test, y_train, y_test = np.load("./kfood/kfood.npy", allow_pickle=True)
    X_train, X_test, y_train, y_test = xy
    # X_train = X_train[:133]
    # X_test = X_test[:133]
    # X_train = X_train.reshape(-1, 64, 64, 3) / 255
    # X_test = X_test.reshape(-1, 64, 64, 3) / 255
    X_train = X_train.astype("float").reshape(-1, 64, 64, 3) / 256
    X_test = X_test.astype("float").reshape(-1, 64, 64, 3) / 256
    y_train = np_utils.to_categorical(y_train, 6)
    y_test = np_utils.to_categorical(y_test, 6)
    return X_train, X_test, y_train, y_test
#%%
# 모델 구성
def build_model(in_shape):
    model = Sequential()
    model.add(Convolution2D(32, 3, padding="same", input_shape=in_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(padding="same"))
    # model.add(Dropout(0.25))
    model.add(Convolution2D(64, 3, padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(128, 3))
    model.add(MaxPooling2D(padding="same"))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(6))
    model.add(Activation("softmax"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
#%%
def model_train(x, y):
    model = build_model(x.shape[1:])
    model.fit(x, y, batch_size=32, epochs=30)
    return model
#%%
def model_eval(model, x, y):
    score = model.evaluate(x, y)
    print("loss = ", score[0])
    print("accuracy = ", score[1])
#%%
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_datasets()
    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)

    # model.save("./kfood/kfood_model.h5")

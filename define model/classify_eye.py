import random, cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

#%% 데이터 로드
imageSize = 28
X = []
Y = []

# 데이터 로드
closed_data_size = 1234
open_data_size = 1234

data_idx = []
for i in range(closed_data_size):
    data_idx.append(i)
for i in range(open_data_size):
    data_idx.append(-i)
random.shuffle(data_idx)

closed_path = './yawn/dataset_new/train/closed2/'
open_path = './yawn/dataset_new/train/open2/'

for i in data_idx:
    # 눈감
    if i > 0:
        img = cv2.imread(closed_path+str(i)+'.jpg')
        Y.append(1)
    # 안눈감
    elif i < 0:
        img = cv2.imread(open_path+str(-i)+'.jpg')
        Y.append(0)
    else:
        continue
    img = cv2.resize(img, (imageSize,imageSize))
    X.append(img/255)

X = np.array(X)
Y = np.array(Y)

#%% 모델 정의

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(imageSize, imageSize, 3)))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Dropout(0.2,))

model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 학습
X_train = X[:2000]
X_val = X[2000:]
Y_train = Y[:2000]
Y_val= Y[2000:]

trained = model.fit(X_train, Y_train, epochs=15, validation_data=(X_val, Y_val))


#%% 모델 저장

model.save('eye.h5')


#%% 시각화

import matplotlib.pyplot as plt

# 정확도
plt.plot(trained.history['accuracy'])
plt.plot(trained.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validataion'], loc='upper left')
plt.show()

# 손실
plt.plot(trained.history['loss'])
plt.plot(trained.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

model.summary()
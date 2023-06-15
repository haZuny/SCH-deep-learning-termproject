import random, cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

#%% 데이터 로드
imageSize = 150
X = []
Y = []

# 데이터 로드
yawn_data_size = 1376
no_yawn_data_size = 1231

data_idx = []
for i in range(yawn_data_size):
    data_idx.append(i)
for i in range(no_yawn_data_size):
    data_idx.append(-i)
random.shuffle(data_idx)

yawn_path = './../img_data/yawn/dataset_new/train/yawn2/'
no_yawn_path = './../img_data/yawn/dataset_new/train/no_yawn2/'

# 얼굴 객체 추출 정의
face_classifier = cv2.CascadeClassifier('.\haarcascade_frontalface_alt2.xml')

for i in data_idx:
    isFindFace = False
    # 하품
    if i > 0:
        Y.append(1)
        # 이미지 읽어와서 얼굴 객체 탐지
        img = cv2.imread(yawn_path+str(i)+'.jpg')
        faces = face_classifier.detectMultiScale(img, minSize=(100, 100), maxSize=(400, 400))
    # 안하품
    elif i < 0:
        # 이미지 읽어와서 얼굴 객체 탐지
        img = cv2.imread(no_yawn_path+str(-i)+'.jpg')
        faces = face_classifier.detectMultiScale(img, minSize=(100, 100), maxSize=(400, 400))
        Y.append(0)
    else:
        continue
    if len(faces)>= 1:
        x, y, w, h = faces[0]
        img = cv2.resize(img[y:y+h, x:x+w], (imageSize,imageSize))
        X.append(img/255)
    else:
        Y.pop(-1)

X = np.array(X)
Y = np.array(Y)

print(X.shape, Y.shape)

#%% 모델 정의

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(imageSize, imageSize, 3)))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Dropout(0.2,))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 학습
X_train = X[:1000]
X_val = X[1000:]
Y_train = Y[:1000]
Y_val= Y[1000:]

trained = model.fit(X_train, Y_train, epochs=8, validation_data=(X_val, Y_val))


#%% 모델 저장

model.save('yawn.h5')


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
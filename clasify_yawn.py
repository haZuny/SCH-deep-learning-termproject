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

yawn_path = './yawn/dataset_new/train/yawn2/'
no_yawn_path = './yawn/dataset_new/train/no_yawn2/'

for i in data_idx:
    # 하품
    if i > 0:
        img = cv2.imread(yawn_path+str(i)+'.jpg')
        Y.append(1)
    # 안하품
    elif i < 0:
        img = cv2.imread(no_yawn_path+str(-i)+'.jpg')
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
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 학습
X_train = X[:2000]
X_val = X[2000:]
Y_train = Y[:2000]
Y_val= Y[2000:]

trained = model.fit(X_train, Y_train, epochs=25, validation_data=(X_val, Y_val))

#%%


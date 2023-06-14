import random, cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 이미지 파일로 예측
def predict_image(file_name):
    
    imageSize = 150
    model = tf.keras.models.load_model('./yawn.h5')

    img = cv2.imread(file_name)
    img = cv2.resize(img, (imageSize,imageSize))/255
    x = np.array([img])
    predict = model.predict(x)
    print(predict)

    if predict[0][0] >= 0.75:
        return 1
    return 0


# 이미지 불러오기
#path = './yawn_test/12.png'
#print(predict_image(path))


imageSize = 150
model = tf.keras.models.load_model('./yawn.h5')
# 넘파이 배열로 예측
def predict_frame(frame):
    
    img = cv2.resize(frame, (imageSize,imageSize))/255
    x = np.array([img])
    predict = model.predict(x)

    if predict[0][0] >= 0.7:
        return 1
    return 0
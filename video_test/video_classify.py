import sys
sys.path.append('./../use_model')
import load_yawn, load_eye
import cv2, time
import tensorflow as tf


# 얼굴, 눈 객체 추출 정의
face_classifier = cv2.CascadeClassifier('.\haarcascade_frontalface_alt2.xml')
eye_classifier = cv2.CascadeClassifier('.\haarcascade_eye.xml')

# 딥러닝 모델 로드
eyeModel = tf.keras.models.load_model('./eye.h5')

prev_time = 0
FPS = 20 # 초당 프레임(낮을수록 부담X but 끊김)
waitTime = 10   # 화면 전화 대기 시간

# 불러올 비디오 영상
capture = cv2.VideoCapture('./test-video1.mp4')

while capture.isOpened():

    run, frame = capture.read()
    
    current_time = time.time() - prev_time
    
    if run and current_time > 1./FPS:
        
        prev_time = time.time()
    
        # 얼굴 객체 탐지
        faces = face_classifier.detectMultiScale(frame, minSize=(100, 100), maxSize=(400, 400))
        
        # 영상 받아오기
        for (x, y, w, h) in faces:
            # 얼굴 빨간색 사각형 그리기
            isYawn = load_yawn.predict_frame(frame[y:y+h, x:x+w])
            if isYawn == 1:
                yawnText = 'yawn'
            else:
                yawnText = 'no-yawn'
            
            cv2.rectangle(frame, (x, y, w, h), (255, 0, 255), 2) 
            
            # 하품 여부 라벨링
            cv2.putText(img=frame,text=yawnText, org=(x, y-10), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale = 2, color=(255,0,255), thickness=2, bottomLeftOrigin=False)
    
            #눈 검출
            #face_half = frame[y:y + h // 2, x:x + w] #얼굴 위쪽에서만 (빨리 찾기 위함)
            #eyes = eye_classifier.detectMultiScale(face_half)
            
            eyes = []
            eyeW = w//5
            eyeH = h//4
            eyes.append((x+eyeW, y+eyeH, eyeW, eyeH))
            eyes.append((x+eyeW*3, y+eyeH, eyeW, eyeH))
    
            for (ex, ey, ew, eh) in eyes:
                # 눈 감은 여부 탐색
                #eyeClose = load_eye.predict_frame(face_half[ey:ey+eh, ex:ex+ew], eyeModel)
                eyeClose = load_eye.predict_frame(frame[ey:ey+eh, ex:ex+ew], eyeModel)
                
                if eyeClose == 0:
                    eyeText = 'open'
                elif eyeClose == 1:
                    eyeText = 'close'
                else:
                    eyeText = 'ambiguous'
                
                # 눈 라벨링
                #cv2.putText(img=face_half, text=eyeText, org=(ex, ey-10), fontFace=cv2.FONT_HERSHEY_PLAIN,
                #            fontScale = 2, color=(255,0,0), thickness=2, bottomLeftOrigin=False)
                cv2.putText(img=frame, text=eyeText, org=(ex, ey-10), fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale = 2, color=(0,0,255), thickness=2, bottomLeftOrigin=False)
                # 눈 파란색 사각형 그리기
                cv2.rectangle(frame, (ex, ey, ew, eh), (0, 0, 255), 2) 
                
        # 이미지 띄우기
        cv2.imshow('video', frame)
        if cv2.waitKey(waitTime) & 0xFF == ord('q'):
            break
        
    if run == False:
        break
                
        
        

capture.release()
cv2.destroyAllWindows()
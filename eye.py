import cv2
import load_yawn
import time

# 객체 생성 및 학습 데이터 불러오기
#얼굴 검출
face_classifier = cv2.CascadeClassifier('.\haarcascade_frontalface_alt2.xml')
#눈 검출
eye_classifier = cv2.CascadeClassifier('.\haarcascade_eye.xml')

capture = cv2.VideoCapture('./test-video3.mp4')

prev_time = 0
FPS = 15

while capture.isOpened():
    run, frame = capture.read()
    
    current_time = time.time() - prev_time
    
    if run and current_time > 1./FPS:
        prev_time = time.time()
    
        # 멀티 스케일 객체 검출 함수
        faces = face_classifier.detectMultiScale(frame, minSize=(100, 100), maxSize=(400, 400))
        
        # 영상 받아오기
        for (x, y, w, h) in faces:
            # 얼굴 빨간색 사각형 그리기
            cv2.rectangle(frame, (x, y, w, h), (255, 0, 255), 2) 
    
            #눈 검출
            face_half = frame[y:y + h // 2, x:x + w] #위 화면에서만 (빨리 찾기 위함)
            eyes = eye_classifier.detectMultiScale(face_half)
            print(len(eyes))
    
            # 눈 파란색 사각형 그리기
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_half, (ex, ey, ew, eh), (255, 0, 0), 2) 
               
               
        
        cv2.imshow('video', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
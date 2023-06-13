import cv2
import load_yawn
import time

capture = cv2.VideoCapture('./test-video3.mp4')

prev_time = 0
FPS = 15

while capture.isOpened():
    run, frame = capture.read()
    
    current_time = time.time() - prev_time
    
    if run and current_time > 1./FPS:
        prev_time = time.time()
    
        if load_yawn.predict_frame(frame) == 1:
            isYawnText = 'yawn'
        else:
            isYawnText = "no-yawn"
        
        cv2.putText(img=frame,text=isYawnText, org=(100,100), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale = 7, color=(0,0,255), thickness=10, bottomLeftOrigin=False)
        
        cv2.imshow('video', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
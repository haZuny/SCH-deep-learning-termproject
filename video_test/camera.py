import cv2, time
 
cap = cv2.VideoCapture(0)
# cap으로 영상을 불러옵니다.
 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
# 영상 프레임 사이즈를 결정합니다.
 
if cap.isOpened():
# 만약 카메라가 실행되고 있다면,
    ret, a = cap.read()
    # ret: True False value입니다.
    # a: 영상 프레임을 읽어옵니다.
 
    while ret:
    # 제대로 카메라를 불러왔다면~ 반복문을 실행합니다. 
        ret, a = cap.read()
        cv2.imshow("camera", a)
        # 이미지를 보여주는 방식과 같습니다.
 
        if cv2.waitKey(1) & 0xFF == 27:
            break123
        # 종료 커맨드.
 
cap.release()
cv2.destroyAllWindows()
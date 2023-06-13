import cv2, csv

# 파일 이름 목록 얻기
fileNames = set()
labelFile = open('train.csv', 'r', encoding = 'utf-8')
labelReader = csv.reader(labelFile)
for line in labelReader:
    fileNames.add(line[0])
labelFile.close()

imgPath = './Images/'

# 이미지 반전 및 저장
for fileName in fileNames:
    # csv의 첫번째 row 패스
    if fileName == "Id":
        continue
    img = cv2.imread(imgPath+fileName+'.jpg')
    img_flip_ud = cv2.flip(img, 0)  # 상하 반전
    cv2.imwrite(imgPath+fileName+'_up_down.jpg', img_flip_ud)
    img_flip_lr = cv2.flip(img, 1)  # 좌우 반전
    cv2.imwrite(imgPath+fileName+'_left_right.jpg', img_flip_lr)
    img_flip_udlr = cv2.flip(img, -1)    # 상하좌우 반전
    cv2.imwrite(imgPath+fileName+'_up_down_left_rignt.jpg', img_flip_udlr)


print("데이터 증가 완료")
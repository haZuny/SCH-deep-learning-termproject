import cv2, os
 
#%% 하품
# 경로
path = './yawn/dataset_new/train/yawn/'

# 파일 목록
file_list = os.listdir(path)

# 저장 경로
save_path = './yawn/dataset_new/train/yawn2/'
img_cnt = len(file_list)

for i, fn in enumerate(file_list):
    img = cv2.imread(path+fn)
    flipImg = cv2.flip(img, 1)
    cv2.imwrite(save_path+str(i)+'.jpg', img)
    cv2.imwrite(save_path+str(img_cnt+i)+'.jpg', flipImg)

print('저장 완료')


#%% 안하품
# 경로
path = './yawn/dataset_new/train/no_yawn/'

# 파일 목록
file_list = os.listdir(path)

# 저장 경로
save_path = './yawn/dataset_new/train/no_yawn2/'
img_cnt = len(file_list)

for i, fn in enumerate(file_list):
    img = cv2.imread(path+fn)
    flipImg = cv2.flip(img, 1)
    cv2.imwrite(save_path+str(i)+'.jpg', img)
    cv2.imwrite(save_path+str(img_cnt+i)+'.jpg', flipImg)

print('저장 완료')
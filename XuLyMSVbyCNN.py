from imutils import contours
import cv2
import imutils
import numpy as np
from PIL import Image,ImageTk
from imutils.perspective import four_point_transform
#Tính thời gian chạy
import time
start_time = time.time()

from tensorflow.keras.models import load_model

mid_hour=time.time()

img=cv2.imread('TEST_50C/AnhSo1.jpg')
height, width= img.shape[:2]



# new_width = int(width / 2)
# new_height = int(height /2)
#
# rz_img=cv2.resize(img,(new_width,new_height))



grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blubled=cv2.GaussianBlur(grey,(3,3),0)
edged=cv2.Canny(blubled,0,100)

Vien= cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
Vien=imutils.grab_contours(Vien)

Khung=[]
if len(Vien)>0:

        Vien = sorted(Vien, key=cv2.contourArea, reverse=True)
        # docCnt=None
        for c in Vien:

            approx = cv2.approxPolyDP(c,0.02*cv2.arcLength(c,True),True)
            area =cv2.contourArea(c)
            if len(approx)==4 and area>20000 :
                # docCnt=approx
                Khung.append(c)
print(len(Khung))

#Lọc ra 2 cột mã sinh viên và mã đề
msvAndIdtest=[]
for k in Khung:
    x,y,w,h=cv2.boundingRect(k)
    if h-1.5*w>0 and h*w<150000:
        msvAndIdtest.append(k)
cv2.drawContours(img,msvAndIdtest,-1,(0,255,0),3)
#Cắt lấy vùng msv
docCnt1=approx = cv2.approxPolyDP(msvAndIdtest[0],0.02*cv2.arcLength(msvAndIdtest[0],True),True)
paper1= four_point_transform(img, docCnt1.reshape(4,2))
warped1=four_point_transform(grey ,docCnt1.reshape(4,2))
#Cắt lấy vùng id test
docCnt2= cv2.approxPolyDP(msvAndIdtest[1],0.02*cv2.arcLength(msvAndIdtest[1],True),True)
paper2= four_point_transform(img, docCnt2.reshape(4,2))
warped2=four_point_transform(grey ,docCnt2.reshape(4,2))

#Lọc ra 2 cột đáp án
quesBox=[]
for k in Khung:

        x, y, w, h = cv2.boundingRect(k)
        if h - 1.5 * w > 0 and h * w > 150000:
            quesBox.append(k)

quesBox=contours.sort_contours(quesBox,method="left-to-right")[0]

response_msv = []
x_curr = 0
y_curr = 0
h_msv, w_msv, c_msv = paper1.shape
print(paper1.shape)
# Biến lưu tọa độ của tất cả các hình chữ nhật bao quanh ô tròn
ToaDo=[]



# Chia lưới cho MSV
for col in range(6):
    max_white_pixels_msv = 0
    selected_digit_msv = None

    for row in range(10):
        # Vùng chọn cho MSV
        x_msv = x_curr + col * (w_msv // 6)
        y_msv = y_curr + row * (h_msv // 10)

        # Cắt phần con của vùng chọn, + thêm vài pixel để bỏ đi viền
        sub_region_msv = warped1[y_msv + 10: y_msv + (h_msv // 10), x_msv: x_msv + (w_msv // 6)]

        # Làm mờ và phát hiện cạnh
        sub_region_msv_blur = cv2.GaussianBlur(sub_region_msv, (3, 3), 0)
        thresh = cv2.threshold(sub_region_msv_blur, 25, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh=cv2.resize(thresh,(28,28),cv2.INTER_AREA)
        thresh=thresh.reshape((28,28,1))
        ToaDo.append(thresh)




response_idt = []
x_curr = 0
y_curr = 0
h_idt, w_idt, c_idt = paper2.shape
print(paper2.shape)
# Biến lưu tọa độ của tất cả các hình chữ nhật bao quanh ô tròn
ToaDo1=[]

# Chia lưới cho id test
for col in range(3):
    max_white_pixels_msv = 0
    selected_digit_msv = None

    for row in range(10):
        # Vùng chọn cho MSV
        x_idt = x_curr + col * (w_idt // 3)
        y_idt = y_curr + row * (h_idt // 10)

        # Cắt phần con của vùng chọn, + thêm vài pixel để bỏ đi viền
        sub_region_msv = warped2[y_idt + 10: y_idt + (h_idt // 10), x_idt: x_idt + (w_idt // 3)]

        # Làm mờ và phát hiện cạnh
        sub_region_msv_blur = cv2.GaussianBlur(sub_region_msv, (3, 3), 0)
        thresh = cv2.threshold(sub_region_msv_blur, 25, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(thresh, (28, 28), cv2.INTER_AREA)
        thresh = thresh.reshape((28, 28, 1))
        ToaDo1.append(thresh)


def get_answers(list_answers):
    response=[]

    model = load_model('ChoiceDitector1.keras')
    list_answers = np.array(list_answers)
    scores = model.predict_on_batch(list_answers / 255.0)

    for idx, score in enumerate(scores):
        mod = idx % 10

        # print(idx)
        # score [unchoiced_cf, choiced_cf]
        if score[1] > 0.9:  # choiced confidence score > 0.9
            response.append(mod)
            print(f'idx = {idx} , mod = {mod}, %={score[1]}')

    return response

response_msv=get_answers(ToaDo)
print("Mã sinh viên",response_msv)
response_idt=get_answers(ToaDo1)
print("Mã đề ",response_idt)

end_time = time.time()

# Thời gian thực thi
print(f"Thời gian tải mô hình: {mid_hour - start_time} giây")
print(f"Thời gian chạy: {end_time - start_time} giây")

paper1=cv2.resize(paper1,(149,372))
paper2=cv2.resize(paper2,(66,372))
img=cv2.resize(img,(786,1118))

cv2.imshow('img,',img)
cv2.imshow('msv',paper1)
cv2.imshow('idt',paper2)
cv2.waitKey()
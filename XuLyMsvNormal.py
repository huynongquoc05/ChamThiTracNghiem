from imutils import contours
import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform
import time
start_time = time.time()


img=cv2.imread('TEST_50C/AnhSo12.jpg')
height, width= img.shape[:2]

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
cv2.imwrite('saveimg.jpg',img)
WhitePixelDif=[]
#Cắt lấy vùng msv
docCnt1=approx = cv2.approxPolyDP(msvAndIdtest[0],0.02*cv2.arcLength(msvAndIdtest[0],True),True)
paper1= four_point_transform(img, docCnt1.reshape(4,2))
warped1=four_point_transform(grey ,docCnt1.reshape(4,2))
#Cắt lấy vùng id test
docCnt2= cv2.approxPolyDP(msvAndIdtest[1],0.02*cv2.arcLength(msvAndIdtest[1],True),True)
paper2= four_point_transform(img, docCnt2.reshape(4,2))
warped2=four_point_transform(grey ,docCnt2.reshape(4,2))

response_msv = []
x_curr = 0
y_curr = 0
h_msv, w_msv, c_msv = paper1.shape
print(paper1.shape)
# Biến lưu tọa độ của tất cả các hình chữ nhật bao quanh ô tròn


# Chia lưới cho MSV
for col in range(6):
    max_white_pixels_msv = 0
    selected_digit_msv = None
    white_pixel_list = []
    for row in range(10):
        # Vùng chọn cho MSV
        x_msv = x_curr + col * (w_msv // 6)
        y_msv = y_curr + row * (h_msv // 10)

        # Cắt phần con của vùng chọn, + thêm vài pixel để bỏ đi viền
        sub_region_msv = warped1[y_msv + 10: y_msv + (h_msv // 10), x_msv: x_msv + (w_msv // 6)]

        # Làm mờ và phát hiện cạnh
        sub_region_msv_blur = cv2.GaussianBlur(sub_region_msv, (3, 3), 0)
        thresh = cv2.threshold(sub_region_msv_blur, 25, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours_msv = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_msv = imutils.grab_contours(contours_msv)

        # Biến để chứa contour hợp lệ
        curr_msv = []

        for v in contours_msv:
            peri = cv2.arcLength(v, True)
            approx = cv2.approxPolyDP(v, 0.01 * peri, True)

            # Lọc các contour có hình dạng tròn
            if len(approx) > 6:
                x1, y1, w1, h1 = cv2.boundingRect(v)
                ar = w1 / float(h1)
                area = cv2.contourArea(v)

                # Lọc các vùng có diện tích nhỏ hoặc không đủ lớn để là hình tròn hợp lệ
                if 50 < area < 10000 and 0.9 <= ar <= 1.1:
                    circularity = (4 * np.pi * area) / (peri ** 2)

                    # Lọc theo độ tròn
                    if circularity > 0.7:
                        curr_msv.append(v)
     # Nếu tìm thấy contour hợp lệ, đếm số lượng pixel trắng trong vùng đó
        if curr_msv:
            # Tạo mặt nạ để vẽ contour
            mask_msv = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask_msv, curr_msv, -1, 255, -1)

            # Áp dụng bitwise_and để chỉ giữ lại các pixel bên trong ô tròn
            mask_msv = cv2.bitwise_and(thresh, thresh, mask=mask_msv)


            white_pixels_msv = cv2.countNonZero(mask_msv)
            white_pixel_list.append(white_pixels_msv)
            # print(f"Col {col}, Row {row}, White pixels: {white_pixels_msv}")  # Debugging output
            # cv2.imshow('mask_msv',mask_msv)
            # cv2.waitKey()
            # # Điều kiện chọn ô có số lượng pixel trắng ít nhất
            if white_pixels_msv > max_white_pixels_msv:
                max_white_pixels_msv = white_pixels_msv
                selected_digit_msv = row
                selected_region_coords = (x_msv, y_msv, x_msv + (w_msv // 6), y_msv + (h_msv // 10))

    # Ghi lại kết quả
    print('selected_digit_msv = ',selected_digit_msv)
    response_msv.append(selected_digit_msv)
    white_pixel_list.sort(reverse=True)
    print(f"Số pixel trắng vùng được chọn: {white_pixel_list[0]}")
    print(f"Số pixel trắng lớn nhất của vùng không được chọn: {white_pixel_list[1]}")
    diffe = white_pixel_list[0] / white_pixel_list[1]
    print(f"Mức chênh lệch: {diffe}")
    WhitePixelDif.append(diffe)
    if selected_region_coords:
        cv2.rectangle(paper1, (selected_region_coords[0], selected_region_coords[1]),
                      (selected_region_coords[2], selected_region_coords[3]), (0, 255, 0), 2)



#id_test
response_idt = []
x_curr = 0
y_curr = 0
h_idt, w_idt, c_idt = paper2.shape
print(paper2.shape)
# Biến lưu tọa độ của tất cả các hình chữ nhật bao quanh ô tròn


# Chia lưới cho id test
for col in range(3):
    max_white_pixels_msv = 0
    selected_digit_msv = None
    white_pixel_list = []
    for row in range(10):
        # Vùng chọn cho MSV
        x_idt = x_curr + col * (w_idt // 3)
        y_idt = y_curr + row * (h_idt // 10)

        # Cắt phần con của vùng chọn, + thêm vài pixel để bỏ đi viền
        sub_region_msv = warped2[y_idt + 10: y_idt + (h_idt // 10), x_idt: x_idt + (w_idt // 3)]

        # Làm mờ và phát hiện cạnh
        sub_region_msv_blur = cv2.GaussianBlur(sub_region_msv, (3, 3), 0)
        thresh = cv2.threshold(sub_region_msv_blur, 25, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Tìm các đường viền trong vùng con
        contours_idTest = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_idTest = imutils.grab_contours(contours_idTest)

        # Biến để chứa contour hợp lệ
        curr_idTest = []

        for v in contours_idTest:
            peri = cv2.arcLength(v, True)
            approx = cv2.approxPolyDP(v, 0.01 * peri, True)

            # Lọc các contour có hình dạng tròn
            if len(approx) > 6:
                x1, y1, w1, h1 = cv2.boundingRect(v)
                ar = w1 / float(h1)
                area = cv2.contourArea(v)

                # Lọc các vùng có diện tích nhỏ hoặc không đủ lớn để là hình tròn hợp lệ
                if 50 < area < 10000 and 0.9 <= ar <= 1.1:
                    circularity = (4 * np.pi * area) / (peri ** 2)

                    # Lọc theo độ tròn
                    if circularity > 0.7:
                        curr_idTest.append(v)



        # Nếu tìm thấy contour hợp lệ, đếm số lượng pixel trắng trong vùng đó
        if curr_idTest:
            # Tạo mặt nạ để vẽ contour
            mask_idt = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask_idt, curr_idTest, -1, 255, -1)

            # Áp dụng bitwise_and để chỉ giữ lại các pixel bên trong ô tròn
            mask_idt = cv2.bitwise_and(thresh, thresh, mask=mask_idt)


            white_pixels_msv = cv2.countNonZero(mask_idt)
            white_pixel_list.append(white_pixels_msv)
            # print(f"Col {col}, Row {row}, White pixels: {white_pixels_msv}")  # Debugging output
            # cv2.imshow('mask_msv',mask_msv)
            # cv2.waitKey()
            # # Điều kiện chọn ô có số lượng pixel trắng ít nhất
            if white_pixels_msv > max_white_pixels_msv:
                max_white_pixels_msv = white_pixels_msv
                selected_digit_msv = row
                selected_region_coords = (x_idt, y_idt, x_idt + (w_idt // 3), y_idt + (h_idt // 10))

    # Ghi lại kết quả
    response_idt.append(selected_digit_msv)
    white_pixel_list.sort(reverse=True)
    print(f"Số pixel trắng vùng được chọn: {white_pixel_list[0]}")

    print(f"Số pixel trắng lớn nhất của vùng không được chọn: {white_pixel_list[1]}")
    diffe = white_pixel_list[0] / white_pixel_list[1]
    print(f"Mức chênh lệch: {diffe}")
    WhitePixelDif.append(diffe)
    if selected_region_coords:
        cv2.rectangle(paper2, (selected_region_coords[0], selected_region_coords[1]),
                      (selected_region_coords[2], selected_region_coords[3]), (0, 255, 0), 2)
print("Mã sinh viên",response_msv)
print("Mã đề",response_idt)

end_time = time.time()

# Thời gian thực thi
print(f"Thời gian chạy: {end_time - start_time} giây")

paper1=cv2.resize(paper1,(149,372))
paper2=cv2.resize(paper2,(66,372))
img=cv2.resize(img,(786,1118))

cv2.imshow('img,',img)
cv2.imshow('msv',paper1)
cv2.imshow('idt',paper2)
cv2.waitKey()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(WhitePixelDif, marker='o', linestyle='-', color='b')

# Thêm tiêu đề và nhãn cho trục
plt.title('Mức Chênh Lệch Pixel Trắng Giữa Các Ô', fontsize=14)
plt.xlabel('Vị trí (chỉ số ô)', fontsize=12)
plt.ylabel('Mức Chênh Lệch (tỷ lệ)', fontsize=12)

# Hiển thị biểu đồ
plt.grid(True)
plt.show()
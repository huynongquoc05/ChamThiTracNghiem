import time
beginTime=time.time()
import cv2
import numpy as np
import imutils

from imutils import contours

img = cv2.imread('Test/MauGiay3.png')
img=cv2.resize(img,(786,1118))
# scaleh=img.shape[0]*3
# scalew=img.shape[1]*3

print(img.shape)
WhitePixelDif=[]
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grey, (3, 3), 0)
sub_region_msv_blur = cv2.bilateralFilter(grey, d=3, sigmaColor=50, sigmaSpace=50)
edged = cv2.Canny(blurred, 0, 75)

# Tìm đường viền 4 cột chứa các câu trả lời
# cv2.imshow('thresh',thresh)
Vien = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
Vien = imutils.grab_contours(Vien)

boxAnns = []
if len(Vien) > 0:
    Vien = sorted(Vien, key=cv2.contourArea, reverse=True)
    for v in Vien:
        peri = cv2.arcLength(v, True)
        approx = cv2.approxPolyDP(v, 0.01 * peri, True)
        # if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(v)
        if 100000 < w * h<300000:
            boxAnns.append(v)

boxAnns = contours.sort_contours(boxAnns, 'left-to-right')[0]
cv2.drawContours(img, boxAnns, -1, (255, 0, 0), 2)
print(len(boxAnns))
height, width = img.shape[:2]
mid = int(height * 0.3)
botimg = img[mid:height, :]
cv2.imshow('botimg1', botimg)

corr=0
question_index = 1
response = {}
for b in boxAnns:
    x, y, w, h = cv2.boundingRect(b)
    h_con = h // 6
    y_con = y

    for i in range(6):
    # Xác định vùng hình chữ nhật con
        VungChon0 = img[y_con:y_con + h_con, x:x + w]

        VungChon = grey[y_con:y_con + h_con, x:x + w]
        #Phóng to vùng chọn để dễ dàng xử lý
        scaleh=VungChon0.shape[0]*3
        scalew=VungChon0.shape[1]*3
        VungChonScale0=cv2.resize(VungChon0,(scalew,scaleh))
        VungChonScale=cv2.resize(VungChon,(scalew,scaleh))
        VungChonblr = cv2.GaussianBlur(VungChonScale, (3, 3), 0)

        thresh = cv2.threshold(VungChonblr, 25, 200, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # Tìm đường viền của các ô trắc nghiệm trong mỗi hình chữ nhật con
        vienOTracN = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vienOTracN = imutils.grab_contours(vienOTracN)
        questionVien = []
        for v in vienOTracN:
            peri = cv2.arcLength(v, True)
            approx = cv2.approxPolyDP(v, 0.01 * peri, True)
            if len(approx)> 6 :
                x1, y1, w1, h1 = cv2.boundingRect(v)
                ar = w1 / float(h1)
                area = cv2.contourArea(v)
                if area == 0:
                    continue  # Bỏ qua các đối tượng không có diện tích
                if area < 100 :
                    continue
                circularity = (4 * np.pi * area) / (cv2.arcLength(v, True) ** 2)

                # Lọc các ô trắc nghiệm có dạng hình tròn với độ tròn > 0.7
                if w1 >= 10 and h1 >= 10 and 0.8 <= ar <= 1.2 and circularity > 0.7 :
                    questionVien.append(v)

        questionVien = contours.sort_contours(questionVien, method="top-to-bottom")[0]
        print(len(questionVien))

        # # Duyệt lấy câu trả lời (Đáp án)
        for (q, i) in enumerate(np.arange(0, len(questionVien), 4)):
            vienOTracN = contours.sort_contours(questionVien[i:i + 4], method="left-to-right")[0]
            bubbled = None
            white_pixel_list = []
            for (j, c) in enumerate(vienOTracN):
                # Mặt nạ đen
                mask = np.zeros(thresh.shape, dtype='uint8')
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)
                white_pixel_list.append(total)
                # print('question', question_index, 'index', j, 'white pixel', total)
                # Lấy ô ít pixel trắng nhất
                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, j)
            color = (0, 255, 0)
            white_pixel_list.sort(reverse=True)
            print(f"Số pixel trắng vùng được chọn: {white_pixel_list[0]}")
            print(f"Số pixel trắng lớn nhất của vùng không được chọn: {white_pixel_list[1]}")
            diffe=white_pixel_list[0] / white_pixel_list[1]
            print(f"Mức chênh lệch: {diffe}")
            WhitePixelDif.append(diffe)
            ans_j = bubbled[1]  # Lấy chỉ số ô được chọn (đáp án)
            # Chuyển đổi đáp án sang A,B,C,D
            ans_j_convert = None
            if ans_j == 0:
                ans_j_convert = 'A'
            elif ans_j == 1:
                ans_j_convert = 'B'
            elif ans_j == 2:
                ans_j_convert = 'C'
            else:
                ans_j_convert = 'D'
            cv2.drawContours(VungChonScale0, [vienOTracN[ans_j]], -1, color, 5)
            # Thêm chỉ số câu hỏi và đáp án vào từ điển response
            response[question_index] = ans_j_convert
            print(f'Câu {question_index} - {ans_j_convert}')
            question_index += 1

        restoreh = int(VungChonScale0.shape[0] / 3)
        restorew = int(VungChonScale0.shape[1] / 3)
        VungChonScale0 = cv2.resize(VungChonScale0, (restorew, restoreh))

        img[y_con:y_con + h_con, x:x + w] = VungChonScale0
        # cv2.imshow('Vungchon0',img[y_con:y_con + h_con, x:x + w])
        # cv2.waitKey()
        y_con += h_con



print(response)
endTime=time.time()

print(f"Thời gian chạy {endTime-beginTime} giây")

# blurred=cv2.resize(blurred,(786,1118))

topimg=img[0:mid,:]

# cv2.imshow('topimg',topimg)
cv2.imshow('botimg', botimg)
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
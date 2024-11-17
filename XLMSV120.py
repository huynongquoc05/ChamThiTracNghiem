import cv2
import numpy as np
import imutils
from imutils import contours
import time
start_time = time.time()

# Đọc ảnh đầu vào
img = cv2.imread('Test/MauGiay6.png')
img =cv2.resize(img,(786,1118))
# Chuyển đổi sang ảnh xám
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grey, (3, 3), 0)
edged = cv2.Canny(blurred, 75, 200)

# Tìm các đường viền
VienMSV = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
VienMSV = imutils.grab_contours(VienMSV)

# Xác định vùng chứa mã SV
boxMsv = []
if len(VienMSV) > 0:
    VienMSV = sorted(VienMSV, key=cv2.contourArea, reverse=True)
    for v in VienMSV:
        peri = cv2.arcLength(v, True)
        approx = cv2.approxPolyDP(v, 0.01 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(v)
            if 10000 < w * h < 40000 and h - 1.5 * w > 0 and h - 3 * w < 0:
                boxMsv.append((x, y, w, h))

# Sắp xếp và vẽ đường viền của các vùng tìm được
boxMsv = sorted(boxMsv, key=lambda b: b[0])  # Sắp xếp từ trái sang phải
cv2.drawContours(img, [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]) for (x, y, w, h) in boxMsv], -1,
                 (255, 0, 0), 1)


WhitePixelDif=[]
# Xử lý mã sinh viên
msv_box = boxMsv[0]
response_msv = []

# Chia lưới cho MSV
for col in range(6):
    max_white_pixels_msv = 0
    selected_digit_msv = None
    selected_region_coords = None  # Lưu tọa độ hình chữ nhật được chọn
    white_pixel_list = []
    for row in range(10):
        # Vùng chọn cho MSV
        x_msv = msv_box[0] + col * (msv_box[2] // 6)
        y_msv = msv_box[1] + row * (msv_box[3] // 10)
        sub_region_msv = grey[y_msv+2:y_msv + (msv_box[3] // 10)-2, x_msv+2:x_msv + (msv_box[2] // 6)-2]

        # Áp dụng bộ lọc bilateral để giữ chi tiết và làm mịn ảnh
        sub_region_msv_blur = cv2.bilateralFilter(sub_region_msv, d=9, sigmaColor=50, sigmaSpace=50)
        thresh = cv2.threshold(sub_region_msv_blur, 25, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # cv2.imshow('th', thresh)
        # cv2.waitKey()

        white_pixels_msv = cv2.countNonZero(thresh)
        white_pixel_list.append(white_pixels_msv)
        # print(f"Col {col}, Row {row}, White pixels: {white_pixels_msv}")  # Debugging output

        # Điều kiện chọn ô có số lượng pixel trắng lớn nhất
        if white_pixels_msv > max_white_pixels_msv:
            max_white_pixels_msv = white_pixels_msv
            selected_digit_msv = row
            selected_region_coords = (x_msv, y_msv, x_msv + (msv_box[2] // 6), y_msv + (msv_box[3] // 10))

    # Lưu số hàng (digit) đã chọn cho cột MSV này
    response_msv.append(selected_digit_msv)
    print(f"Số pixel trắng vùng được chọn: {white_pixel_list[0]}")
    white_pixel_list.sort(reverse=True)
    print(f"Số pixel trắng lớn nhất của vùng không được chọn: {white_pixel_list[1]}")
    diffe = white_pixel_list[0] / white_pixel_list[1]
    print(f"Mức chênh lệch: {diffe}")
    WhitePixelDif.append(diffe)
    # Vẽ hình chữ nhật vào khu vực được chọn
    if selected_region_coords:
        cv2.rectangle(img, (selected_region_coords[0], selected_region_coords[1]),
                      (selected_region_coords[2], selected_region_coords[3]), (0, 255, 0), 2)

result = ''.join(map(str, response_msv))
cv2.putText(img,result,(msv_box[0]+2, msv_box[1]-15),1,1.5
            ,(0,255,0),2)


VienIDTEST = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
VienIDTEST = imutils.grab_contours(VienIDTEST)

# Xác định vùng chứa mã đề
boxIdtest = []
if len(VienIDTEST) > 0:
    VienIDTEST = sorted(VienIDTEST, key=cv2.contourArea, reverse=True)
    for v in VienIDTEST:
        peri = cv2.arcLength(v, True)
        approx = cv2.approxPolyDP(v, 0.01 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(v)
            if 10000 < w * h < 40000 and h - 3 * w > 0:
                boxIdtest.append((x, y, w, h))

# Sắp xếp và vẽ đường viền của các vùng tìm được
boxIdtest = sorted(boxIdtest, key=lambda b: b[0])  # Sắp xếp từ trái sang phải
cv2.drawContours(img, [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]) for (x, y, w, h) in boxIdtest], -1,
                 (255, 0, 0), 1)

# Xử lý mã đề
if len(boxIdtest) > 0:
    idtest_box = boxIdtest[0]  # Giả định rằng vùng đầu tiên là mã đề

    response_idtest = []

    # Chia lưới cho mã đề
    for col in range(3):
        max_white_pixels_idtest=0
        # min_white_pixels_idtest = 0
        selected_digit_idtest = None
        selected_region_coords_idtest = None
        white_pixel_list=[]
        for row in range(10):

            # Vùng chọn cho mã đề
            x_idtest = idtest_box[0] + col * (idtest_box[2] // 3)
            y_idtest = idtest_box[1] + row * (idtest_box[3] // 10)
            sub_region_idtest = grey[y_idtest+2:y_idtest + (idtest_box[3] // 10)-2, x_idtest+2:x_idtest -2+ (idtest_box[2] // 3)]

            # Áp dụng bộ lọc bilateral để giữ chi tiết và làm mịn ảnh
            sub_region_idtest_blur = cv2.bilateralFilter(sub_region_idtest, d=7, sigmaColor=50, sigmaSpace=50)

            thresh = cv2.threshold(sub_region_idtest_blur, 25, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # cv2.imshow('th', thresh)
            # cv2.waitKey()



            white_pixels_idtest = cv2.countNonZero(thresh)
            # cv2.imshow('mask_idtest', mask_idtest)
            # cv2.waitKey(300)
            # print(f"Col {col}, Row {row}, White pixels: {white_pixels_idtest}")  # Debugging output
            white_pixel_list.append(white_pixels_idtest)
            # Điều kiện chọn ô có số lượng pixel trắng ít nhất
            if white_pixels_idtest > max_white_pixels_idtest:
                max_white_pixels_idtest = white_pixels_idtest
                selected_digit_idtest = row
                selected_region_coords_idtest = (x_idtest, y_idtest, x_idtest + (idtest_box[2] // 3), y_idtest + (idtest_box[3] // 10))

        response_idtest.append(selected_digit_idtest)
        white_pixel_list.sort(reverse=True)
        print(f"Số pixel trắng vùng được chọn: {white_pixel_list[0]}")
        print(f"Số pixel trắng lớn nhất của vùng không được chọn: {white_pixel_list[1]}")
        diffe=white_pixel_list[0]/white_pixel_list[1]
        print(f"Mức chênh lệch: {diffe}")
        WhitePixelDif.append(diffe)
        # Vẽ hình chữ nhật vào khu vực được chọn cho mã đề
        if selected_region_coords_idtest:
            cv2.rectangle(img, (selected_region_coords_idtest[0], selected_region_coords_idtest[1]),
                          (selected_region_coords_idtest[2], selected_region_coords_idtest[3]), (0, 255, 0), 2)
result1 = ''.join(map(str, response_idtest))
idtest_box = boxIdtest[0]
cv2.putText(img, result1, (idtest_box[0] + 2, idtest_box[1] - 15), 1, 1.5
             , (0, 255, 0), 2)

print("Mã sinh viên: ", response_msv)
print("Mã đề: ", response_idtest)

end_time = time.time()

# Thời gian thực thi
print(f"Thời gian chạy: {end_time - start_time} giây")
# Hiển thị kết quả
cv2.imshow('Mã SV', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
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

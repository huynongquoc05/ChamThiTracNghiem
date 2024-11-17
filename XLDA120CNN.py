import time
beginTime=time.time()
import cv2
import numpy as np
import imutils

from imutils import contours

from tensorflow.keras.models import load_model

img = cv2.imread('Test/MauGiay3.png')
img=cv2.resize(img,(786,1118))
# scaleh=img.shape[0]*3
# scalew=img.shape[1]*3

print(img.shape)

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
            # area =cv2.contourArea(v)
            # if area==0:continue
            boxAnns.append(v)

boxAnns = contours.sort_contours(boxAnns, 'left-to-right')[0]
cv2.drawContours(img, boxAnns, -1, (0, 0, 255), 2)
print(len(boxAnns))
# Duyệt từng cột nhóm câu hỏi
#
#   Biến để lưu chỉ số câu hỏi tổng quát
corr=0
question_index = 1
response = {}

ToaDo = []
Predictpcent=[]
# Vòng lặp qua từng boxAnn để xử lý
for b in boxAnns:
    x, y, w, h = cv2.boundingRect(b)
    h_con = h // 6
    y_con = y

    for i in range(6):
        # Xác định vùng hình chữ nhật con
        VungChon0 = img[y_con+int(h_con/20):y_con+h_con-int(h_con/20), x+int(w/6):x + w]
        VungChon = grey[y_con + int (h_con/20): y_con + h_con - int(h_con/20), x + int(w/6): x + w]

        # Tính toán lại chiều cao và chiều rộng
        scaleh = VungChon0.shape[0] * 3
        scalew = VungChon0.shape[1] * 3
        VungChonScale0 = cv2.resize(VungChon0, (scalew, scaleh))
        VungChonScale = cv2.resize(VungChon, (scalew, scaleh))
        VungChonblr = cv2.GaussianBlur(VungChonScale, (3, 3), 0)

        # Tạo ngưỡng ảnh nhị phân
        thresh = cv2.threshold(VungChonblr, 25, 200, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        oneper5h = thresh.shape[0] // 5  # Chiều cao của mỗi ô nhỏ (5 dòng)
        oneper4w = thresh.shape[1] // 4  # Chiều rộng của mỗi ô nhỏ (4 cột)

        # Vòng lặp qua từng dòng và cột để chia lưới
        for row in range(5):  # Duyệt qua 5 dòng
            for col in range(4):  # Duyệt qua 4 cột
                # Xác định tọa độ vùng hình chữ nhật con
                y_conth = row * oneper5h
                x_conth = col * oneper4w

                # Kiểm tra vùng cắt trước khi xử lý
                if y_conth + oneper5h <= thresh.shape[0] and x_conth + oneper4w <= thresh.shape[1]:
                    # Cắt vùng hình chữ nhật con từ ảnh nhị phân
                    VungChonth = thresh[y_conth:y_conth + oneper5h, x_conth:x_conth + oneper4w]

                    # Kiểm tra nếu vùng chọn không rỗng
                    if VungChonth.size > 0:
                        # Resize vùng chọn về kích thước (28, 28)
                        VungChonth_resized = cv2.resize(VungChonth, (28, 28))
                        VungChonth_resized = VungChonth_resized.reshape(28, 28, 1)

                        # Lưu hình chữ nhật con vào danh sách ToaDo
                        ToaDo.append(VungChonth_resized)


                else:
                    print(f"Invalid region at row {row}, col {col}")


        y_con += h_con
print(len(ToaDo))
# Đóng tất cả các cửa sổ hiển thị
def get_answers(list_answers):
    results = {}  # Chuyển sang từ điển thông thường
    # model = CNN_Model('weight.h5').build_model(rt=True)  # Sửa đuôi .h5 thành .keras
    model = load_model('my_model.keras')
    list_answers = np.array(list_answers)
    scores = model.predict_on_batch(list_answers / 255.0)

    for idx, score in enumerate(scores):
        question = idx // 4
        # print(idx)
        # score [unchoiced_cf, choiced_cf]
        if score[1] > 0.9:  # choiced confidence score > 0.9
            Predictpcent.append(score[1])
            if idx%4==0:chosen_answer='A'
            elif idx%4==1:chosen_answer='B'
            elif idx%4==2:chosen_answer='C'
            else:chosen_answer='D'


            results[question + 1]=chosen_answer

    return results


response=get_answers(ToaDo)

print(response)
endTime=time.time()

print(f"Thời gian chạy {endTime-beginTime} giây")
height, width = img.shape[:2]
# blurred=cv2.resize(blurred,(786,1118))
mid = int(height * 0.3)

botimg = img[mid:height, :]
topimg=img[0:mid,:]

# cv2.imshow('topimg',topimg)
cv2.imshow('botimg', botimg)
cv2.waitKey()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(Predictpcent, marker='o', linestyle='-', color='b')

# Thêm tiêu đề và nhãn cho trục
plt.title('Tỉ lệ phần trăm dự đoán của mô hình', fontsize=14)
plt.xlabel('Vị trí (chỉ số ô)', fontsize=12)
plt.ylabel('Tỉ lệ', fontsize=12)

# Hiển thị biểu đồ
plt.grid(True)
plt.show()
import cv2
import numpy as np
import imutils
from imutils import contours
import time
start_time = time.time()

from tensorflow.keras.models import load_model
mid_time=time.time()
# Đọc ảnh đầu vào
img = cv2.imread('NOT IN USE/MauGiay6.png')


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

# Xử lý mã sinh viên
msv_box = boxMsv[0]
response_msv = []
ToaDo=[]
# Chia lưới cho MSV
for col in range(6):

    for row in range(10):
        # Vùng chọn cho MSV
        x_msv = msv_box[0] + col * (msv_box[2] // 6)
        y_msv = msv_box[1] + row * (msv_box[3] // 10)
        sub_region_msv = grey[y_msv:y_msv + (msv_box[3] // 10), x_msv:x_msv + (msv_box[2] // 6)]

        # Áp dụng bộ lọc bilateral để giữ chi tiết và làm mịn ảnh
        sub_region_msv_blur = cv2.bilateralFilter(sub_region_msv, d=9, sigmaColor=50, sigmaSpace=50)

        thresh = cv2.threshold(sub_region_msv_blur, 25, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # cv2.imshow('th',thresh)
        # cv2.waitKey()
        # thresh=cv2.threshold(sub_region_msv_blur,25,200,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
        # Tìm các đường viền trong vùng con
        thresh = cv2.threshold(sub_region_msv_blur, 25, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(thresh, (28, 28), cv2.INTER_AREA)
        thresh = thresh.reshape((28, 28, 1))
        ToaDo.append(thresh)





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
ToaDo1=[]
# Xử lý mã đề
if len(boxIdtest) > 0:
    idtest_box = boxIdtest[0]  # Giả định rằng vùng đầu tiên là mã đề

    response_idtest = []

    # Chia lưới cho mã đề
    for col in range(3):

        for row in range(10):
            # Vùng chọn cho mã đề
            x_idtest = idtest_box[0] + col * (idtest_box[2] // 3)
            y_idtest = idtest_box[1] + row * (idtest_box[3] // 10)
            sub_region_idtest = grey[y_idtest:y_idtest + (idtest_box[3] // 10), x_idtest:x_idtest + (idtest_box[2] // 3)]

            # Áp dụng bộ lọc bilateral để giữ chi tiết và làm mịn ảnh
            sub_region_idtest_blur = cv2.bilateralFilter(sub_region_idtest, d=7, sigmaColor=50, sigmaSpace=50)
            # sub_region_idtest_edged = cv2.Canny(sub_region_idtest_blur, 50, 170)
            thresh = cv2.threshold(sub_region_idtest_blur, 25, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = cv2.resize(thresh, (28, 28), cv2.INTER_AREA)
            thresh = thresh.reshape((28, 28, 1))
            ToaDo1.append(thresh)


def get_answers(list_answers):
    response=[]
    # model = CNN_Model('weight.h5').build_model(rt=True)  # Sửa đuôi .h5 thành .keras
    model = load_model('ChoiceDitector.keras')
    list_answers = np.array(list_answers)
    scores = model.predict_on_batch(list_answers / 255.0)
    for idx, score in enumerate(scores):
        mod = idx % 10

        # print(idx)
        # score [unchoiced_cf, choiced_cf]
        if score[1] > 0.9:  # choiced confidence score > 0.9
            response.append(mod)
            # print(f'idx = {idx} , mod = {mod}')
    return response

response_msv=get_answers(ToaDo)
print("Mã sinh viên: ", response_msv)
response_idtest=get_answers(ToaDo1)
print("Mã đề: ", response_idtest)

end_time = time.time()

# Thời gian thực thi
print(f"Thời gian import load model: {mid_time - start_time} giây")
print(f"Thời gian chạy: {end_time - start_time} giây")
# Hiển thị kết quả
cv2.imshow('Mã SV', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

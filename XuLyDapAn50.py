from imutils import contours
import cv2
import imutils
import numpy as np
from PIL import Image,ImageTk
from imutils.perspective import four_point_transform

img=cv2.imread('TEST_50C/AnhSo1.jpg')
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


#Lọc ra 2 cột đáp án
quesBox=[]
for k in Khung:

        x, y, w, h = cv2.boundingRect(k)
        if h - 1.5 * w > 0 and h * w > 150000:
            quesBox.append(k)

quesBox=contours.sort_contours(quesBox,method="left-to-right")[0]
print("số phần tử trong questBox",len(quesBox))
question_index = 1
response = {}
img_paper=[]
Whitepixel_diffirence=[]
for quest in quesBox:
    docCnt=approx = cv2.approxPolyDP(quest,0.02*cv2.arcLength(quest,True),True)
    paper= four_point_transform(img, docCnt.reshape(4,2))
    warped=four_point_transform(grey ,docCnt.reshape(4,2))
    # cv2.imshow('paper',paper)
    # cv2.waitKey()
    # cv2.imshow('paper',paper[500:,:])

    print(paper.shape)
    h,w,c=paper.shape
    # x, y, w, h = cv2.boundingRect(paper)
    h_con = h // 5
    y_con = 0

    for i in range(5):


        # Xác định vùng hình chữ nhật con
        VungChon0 = paper[y_con+15:y_con + h_con,:]
        cv2.imshow("vc0",VungChon0)
        VungChon = warped[y_con+15:y_con + h_con,:]
        #Phóng to vùng chọn để dễ dàng xử lý
        scaleh=VungChon0.shape[0]*3
        scalew=VungChon0.shape[1]*3
        VungChonScale0=cv2.resize(VungChon0,(scalew,scaleh))
        VungChonScale=cv2.resize(VungChon,(scalew,scaleh))
        VungChonblr = cv2.GaussianBlur(VungChonScale, (3, 3), 0)


        # VungChonedged = cv2.Canny(VungChonblr, 75, 200)
        thresh = cv2.threshold(VungChonblr, 25, 200, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # cv2.imshow('thresh',thresh)
        # cv2.waitKey()
        # cv2.imshow('vccc',VungChonedged)
        vienOTracN = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # # Tìm đường viền của các ô trắc nghiệm trong mỗi hình chữ nhật con
        vienOTracN = imutils.grab_contours(vienOTracN)
        questionVien = []

        for v in vienOTracN:

            peri = cv2.arcLength(v, True)
            approx = cv2.approxPolyDP(v, 0.01 * peri, True)
            if len(approx)> 6 :

                x1, y1, w1, h1 = cv2.boundingRect(v)
                ar = w1 / float(h1)
                area = cv2.contourArea(v)
                if area < 50 and area>10000:
                    continue  #

                circularity = (4 * np.pi * area) / (cv2.arcLength(v, True) ** 2)

                # Lọc các ô trắc nghiệm
                if w1 >= 10 and h1 >= 10 and 0.9 <= ar <= 1.15:
                    questionVien.append(v)
        print(len(questionVien))
        # if len(questionVien)==0:

        questionVien = contours.sort_contours(questionVien, method="top-to-bottom")[0]
        # cv2.drawContours(VungChonScale0,questionVien,-1,(0,255,0),3)
        for (q, i) in enumerate(np.arange(0, len(questionVien), 4)):
            vienOTracN = contours.sort_contours(questionVien[i:i + 4], method="left-to-right")[0]
            bubbled = None
            white_pixel_list = []
            for (j, c) in enumerate(vienOTracN):
                # Mặt nạ đen
                mask = np.zeros(thresh.shape, dtype='uint8')
                cv2.drawContours(mask, [c], -1, 255, -1)
                # cv2.imshow('mask0', mask)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)
                white_pixel_list.append(total)
                # cv2.putText(mask, f'ques {question_index} index {j}, white pixels {total} ', (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
                #
                # cv2.imshow('mask',mask)
                # cv2.waitKey(200)
                # print('question', question_index, 'index', j, 'white pixel', total)
                # Lấy ô nhiều pixel trắng nhất
                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, j)
            color = (0, 0, 255)

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

            # if k == ans_j:
            #     color = (0, 255, 0)
            white_pixel_list.sort(reverse=True)
            print(f"Số pixel trắng vùng được chọn: {white_pixel_list[0]}")

            print(f"Số pixel trắng lớn nhất của vùng không được chọn: {white_pixel_list[1]}")
            difference=white_pixel_list[0] / white_pixel_list[1]
            print(f"Mức chênh lệch: {difference}")
            Whitepixel_diffirence.append(difference)
            cv2.drawContours(VungChonScale0, [vienOTracN[ans_j]], -1, color, 3)
            # Thêm chỉ số câu hỏi và đáp án vào từ điển response
            response[question_index] = ans_j_convert
            print(f'{question_index} - {ans_j_convert}')
            question_index += 1
        # cv2.imshow('thress', thresh)
        restoreh = int(VungChonScale0.shape[0] / 3)
        restorew = int(VungChonScale0.shape[1] / 3)
        VungChonScale0 = cv2.resize(VungChonScale0, (restorew, restoreh))
        # cv2.imshow('vssc0', VungChonScale0)
        # cv2.waitKey(300)
        paper[y_con+15:y_con + h_con, :] = VungChonScale0
        y_con += h_con



    cv2.imwrite("NOT IN USE/maugiaycat.png", paper)
    cv2.imwrite('NOT IN USE/maugiaycat1.png', paper)
    # cv2.imshow('paper', paper)
    img_paper.append(paper)



edged=cv2.resize(edged,(786,1118))

img=cv2.resize(img,(786,1118))
print(response)
# cv2.imshow('cc',edged)
# cv2.imshow('bb',img)

for idx, img in enumerate(img_paper):
    print(img.shape)
    img=cv2.resize(img,(320,750))
    cv2.imshow(f'Image {idx+1}', img)
cv2.waitKey()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(Whitepixel_diffirence, marker='o', linestyle='-', color='b')

# Thêm tiêu đề và nhãn cho trục
plt.title('Mức Chênh Lệch Pixel Trắng Giữa Các Ô', fontsize=14)
plt.xlabel('Vị trí (chỉ số ô)', fontsize=12)
plt.ylabel('Mức Chênh Lệch (tỷ lệ)', fontsize=12)

# Hiển thị biểu đồ
plt.grid(True)
plt.show()





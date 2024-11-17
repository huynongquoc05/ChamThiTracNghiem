from imutils import contours
import cv2
import imutils
import numpy as np
from PIL import Image,ImageTk
from imutils.perspective import four_point_transform

img=cv2.imread('TEST_50C/AnhSo10.jpg')
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
print("số phần tử trong questBox",len(quesBox))
question_index = 1
response = {}
img_paper=[]
for quest in quesBox:
    docCnt=approx = cv2.approxPolyDP(quest,0.02*cv2.arcLength(quest,True),True)
    paper= four_point_transform(img, docCnt.reshape(4,2))
    warped=four_point_transform(grey ,docCnt.reshape(4,2))
    # cv2.imshow('paper',paper[500:,:])

    print(paper.shape)
    h,w,c=paper.shape
    # x, y, w, h = cv2.boundingRect(paper)
    h_con = h // 5
    y_con = 0

    for i in range(5):


        # Xác định vùng hình chữ nhật con
        VungChon0 = paper[y_con+15:y_con + h_con,:]
        VungChon = warped[y_con+15:y_con + h_con,:]
        #Phóng to vùng chọn để dễ dàng xử lý
        scaleh=VungChon0.shape[0]*3
        scalew=VungChon0.shape[1]*3
        VungChonScale0=cv2.resize(VungChon0,(scalew,scaleh))
        VungChonScale=cv2.resize(VungChon,(scalew,scaleh))
        VungChonblr = cv2.GaussianBlur(VungChonScale, (3, 3), 0)


        # VungChonedged = cv2.Canny(VungChonblr, 75, 200)
        thresh = cv2.threshold(VungChonblr, 25, 200, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
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
            for (j, c) in enumerate(vienOTracN):
                # Mặt nạ đen
                mask = np.zeros(thresh.shape, dtype='uint8')
                cv2.drawContours(mask, [c], -1, 255, -1)
                # cv2.imshow('mask0', mask)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)
                # cv2.imshow('mask',mask)
                # cv2.waitKey(200)
                print('question', question_index, 'index', j, 'white pixel', total)
                # Lấy ô ít pixel trắng nhất
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

            cv2.drawContours(VungChonScale0, [vienOTracN[ans_j]], -1, color, 3)
            # Thêm chỉ số câu hỏi và đáp án vào từ điển response
            response[question_index] = ans_j_convert

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

# hmsv,wmsv,cmsv=paper1.shape
# scaleWmsv=int(wmsv*2)
# scaleHmsv=int(hmsv*2)
#
# paper1=cv2.resize(paper1,(scaleWmsv,scaleHmsv))
# warped1=cv2.resize()
#Xử lý mã sinh viên
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

    for row in range(10):
        # Vùng chọn cho MSV
        x_msv = x_curr + col * (w_msv // 6)
        y_msv = y_curr + row * (h_msv // 10)

        # Cắt phần con của vùng chọn, + thêm vài pixel để bỏ đi viền
        sub_region_msv = warped1[y_msv + 10: y_msv + (h_msv // 10), x_msv: x_msv + (w_msv // 6)]

        # Làm mờ và phát hiện cạnh
        sub_region_msv_blur = cv2.GaussianBlur(sub_region_msv, (3, 3), 0)
        thresh = cv2.threshold(sub_region_msv_blur, 25, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Tìm các đường viền trong vùng con
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
            print(f"Col {col}, Row {row}, White pixels: {white_pixels_msv}")  # Debugging output
            # cv2.imshow('mask_msv',mask_msv)
            # cv2.waitKey()
            # # Điều kiện chọn ô có số lượng pixel trắng ít nhất
            if white_pixels_msv > max_white_pixels_msv:
                max_white_pixels_msv = white_pixels_msv
                selected_digit_msv = row
                selected_region_coords = (x_msv, y_msv, x_msv + (w_msv // 6), y_msv + (h_msv // 10))

    # Ghi lại kết quả
    response_msv.append(selected_digit_msv)

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
            print(f"Col {col}, Row {row}, White pixels: {white_pixels_msv}")  # Debugging output
            # cv2.imshow('mask_msv',mask_msv)
            # cv2.waitKey()
            # # Điều kiện chọn ô có số lượng pixel trắng ít nhất
            if white_pixels_msv > max_white_pixels_msv:
                max_white_pixels_msv = white_pixels_msv
                selected_digit_msv = row
                selected_region_coords = (x_idt, y_idt, x_idt + (w_idt // 3), y_idt + (h_idt // 10))

    # Ghi lại kết quả
    response_idt.append(selected_digit_msv)

    if selected_region_coords:
        cv2.rectangle(paper2, (selected_region_coords[0], selected_region_coords[1]),
                      (selected_region_coords[2], selected_region_coords[3]), (0, 255, 0), 2)
print(response_msv)
print(response_idt)

edged=cv2.resize(edged,(786,1118))

img=cv2.resize(img,(786,1118))
print(response)
cv2.imshow('cc',edged)
cv2.imshow('bb',img)
paper1=cv2.resize(paper1,(149,372))
cv2.imshow('msv',paper1)
paper2=cv2.resize(paper2,(66,372))
cv2.imshow('idtest',paper2)
for idx, img in enumerate(img_paper):
    print(img.shape)
    img=cv2.resize(img,(320,750))
    cv2.imshow(f'Image {idx+1}', img)
cv2.waitKey()






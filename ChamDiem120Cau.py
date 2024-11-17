import cv2
import numpy as np
import imutils
import requests
from imutils import contours

#Gọi tới 3 class Sinh vien ,bài kiểm tra và điểm lu trong file models
from Model import SinhVien,QuanLyDiem,BaiKiemTra

# Tạo đối tượng BaiKiemTra với 120 câu hỏi và đáp án ngẫu nhiên
import pyodbc
#lấy dữ liệu bài kiểm tra từ cơ sở dữ liệu:
conn = pyodbc.connect('Server=DESKTOP-AMN3OI8;Database=SinhVienVaDiem;UID=sa;PWD=051120;PORT=1433;DRIVER={SQL Server}')
cursor = conn.cursor()
#lấy dữ liệu sinh viên
cursor.execute('select * from sinhvien')
rows=cursor.fetchall()
listSV=[]
for row in rows:
    sinhvien=SinhVien(ma_sinh_vien=row.ma_sinh_vien.strip(), ten_sinh_vien=row.ten_sinh_vien,
                      lop_hoc_phan=row.lop_hoc_phan)
    listSV.append(sinhvien)
for l in listSV:
    print('\nĐối tượng sinh viên được lấy từ trong cơ sở dữ liệu:')
    l.showInfor()



#ấy dữ liệu bảng quản lý điểm
cursor.execute('select * from quanlydiem')
rows=cursor.fetchall()
listdiem=[]
for row in rows:
    diem=QuanLyDiem(so_thu_tu=row.so_thu_tu,ma_sinh_vien=row.ma_sinh_vien.strip(),
                    ten_sinh_vien=row.ten_sinh_vien,
                    diem_10=row.diem_10_percent,diem_kiem_tra_1=row.diem_kiem_tra_1,
                    diem_kiem_tra_2=row.diem_kiem_tra_2,diem_thi_ket_thuc=row.diem_thi_ket_thuc)
    listdiem.append(diem)

for l in listdiem:
    print("\nĐối tượng điểm được tìm thấy trong cơ sở dữ liệu: ")
    l.showInfor()


# Hàm nhập loại bài kiểm tra với kiểm tra điều kiện
def nhap_loai_bai_kiem_tra():
    # loai_bai_kiem_tra=None
    while True:
        try:
            print("\n1: Bài kiểm tra số 1\n2: Bài kiểm tra số 2\n3: Bài thi kết thúc học phần")
            loai_bai_kiem_tra = int(input("Nhập loại bài kiểm tra (1, 2, hoặc 3): "))
            if loai_bai_kiem_tra in [1, 2, 3]:
                return loai_bai_kiem_tra
            else:
                print("Giá trị không hợp lệ, vui lòng nhập lại.")
        except ValueError:
            print("Giá trị không hợp lệ, vui lòng nhập lại.")
    return loai_bai_kiem_tra

#Phân loại bài kiểm tra muốn lấy
lbkt=nhap_loai_bai_kiem_tra()
if lbkt==1:print("Bạn đang tìm bài kiểm tra số 1")
elif lbkt==2:print("Bạn đang tìm bài kiểm tra số 2")
else:print("Bạn đang tìm bài thi kết thúc học phần")

# Lấy dữ liệu bài kiểm tra
cursor.execute('select * from baikiemtra')
rows = cursor.fetchall()
list_bai_kiem_tra= []

for row in rows:
    text = row.dap_an
    pairs = text.split(',')
    dictionary = {int(k): int(v) for k, v in (pair.split(':') for pair in pairs)}


    if row.loai_bai_kiem_tra==lbkt:
        bkt = BaiKiemTra(ma_de=row.ma_de.strip(), loai_baiKt=row.loai_bai_kiem_tra, so_cau_hoi=row.so_cau_hoi,
                         dap_an=dictionary)
        list_bai_kiem_tra.append(bkt)


if len(list_bai_kiem_tra)==0:print("Không tìm thấy bài kiểm tra")

else:
    print("\nĐối tượng bài kiểm tra được tìm thấy\n")
    for l in list_bai_kiem_tra:
        l.showInfor()

#Đóng kết nối


#Đọc ảnh
img = cv2.imread('NOT IN USE/MauGiay3.png')
img=cv2.resize(img,(786,1118))

height, width = img.shape[:2]

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grey, (3, 3), 0)
edged = cv2.Canny(blurred, 75, 200)


#Hàm xử lý đáp án
response = {}
corectquest={}
Diem=[]
def XulyDapAn(a): # Truyền đối tượng bài kiểm tra
    DapAnMau=a.dap_an #lấy đáp án từ đối tượng
    # Tìm đường viền 4 cột chứa các câu trả lời
    Vien = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Vien = imutils.grab_contours(Vien)

    boxAnns = []
    if len(Vien) > 0:
        Vien = sorted(Vien, key=cv2.contourArea, reverse=True)
        for v in Vien:
            peri = cv2.arcLength(v, True)
            approx = cv2.approxPolyDP(v, 0.01 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(v)
                if 100000 < w * h < 200000:
                    boxAnns.append(v)

    boxAnns = contours.sort_contours(boxAnns, 'left-to-right')[0]
    cv2.drawContours(img, boxAnns, -1, (0, 0, 255), 2)

    # Duyệt từng cột nhóm câu hỏi

    question_index = 1  # Biến để lưu chỉ số câu hỏi tổng quát
    SoCauToiDa=a.so_cau_hoi
    corr=0
    for b in boxAnns:
        x, y, w, h = cv2.boundingRect(b)
        h_con = h // 6
        y_con = y

        for i in range(6):
            if question_index > SoCauToiDa:
                break

            # Xác định vùng hình chữ nhật con
            VungChon0 = img[y_con:y_con + h_con, x:x + w]
            VungChon = grey[y_con:y_con + h_con, x:x + w]
            VungChonblr = cv2.GaussianBlur(VungChon, (3, 3), 0)
            VungChonedged = cv2.Canny(VungChonblr, 75, 200)

            # Tìm đường viền của các ô trắc nghiệm trong mỗi hình chữ nhật con
            vienOTracN = cv2.findContours(VungChonedged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vienOTracN = imutils.grab_contours(vienOTracN)
            questionVien = []

            for v in vienOTracN:

                peri = cv2.arcLength(v, True)
                approx = cv2.approxPolyDP(v, 0.01 * peri, True)
                if len(approx) > 6:

                    x1, y1, w1, h1 = cv2.boundingRect(v)
                    ar = w1 / float(h1)
                    area = cv2.contourArea(v)
                    if area == 0:
                        continue  # Bỏ qua các đối tượng không có diện tích
                    if area < 50:
                        continue
                    circularity = (4 * np.pi * area) / (cv2.arcLength(v, True) ** 2)

                    # Lọc các ô trắc nghiệm có dạng hình tròn với độ tròn > 0.8
                    if w1 >= 10 and h1 >= 10 and 0.9 <= ar <= 1.1 and circularity > 0.75:
                        questionVien.append(v)

            questionVien = contours.sort_contours(questionVien, method="top-to-bottom")[0]

            # Duyệt lấy câu trả lời (Đáp án)
            for (q, i) in enumerate(np.arange(0, len(questionVien), 4)):
                vienOTracN = contours.sort_contours(questionVien[i:i + 4], method="left-to-right")[0]
                bubbled = None
                for (j, c) in enumerate(vienOTracN):
                    # Mặt nạ đen
                    mask = np.zeros(VungChonedged.shape, dtype='uint8')
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    mask = cv2.bitwise_and(VungChonedged, VungChonedged, mask=mask)
                    total = cv2.countNonZero(mask)
                    # print('question', question_index, 'index', j, 'white pixel', total)
                    # Lấy ô ít pixel trắng nhất
                    if bubbled is None or total < bubbled[0]:
                        bubbled = (total, j)
                color = (0, 0, 255)
                k = DapAnMau[question_index]  # lấy đáp án mẫu của câu hỏi
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

                if k == ans_j:
                    color = (0, 255, 0)
                    corectquest[question_index] = ans_j_convert

                cv2.drawContours(VungChon0, [vienOTracN[ans_j]], -1, color, 2)
                # Thêm chỉ số câu hỏi và đáp án vào từ điển response
                response[question_index] = ans_j_convert

                question_index += 1
                if question_index > SoCauToiDa:
                    break  # Dừng nếu đã đạt đến số câu hỏi tối đa

            y_con += h_con


    diem=10/a.so_cau_hoi *len(corectquest)
    diemString=f"Diem so: {diem:.2f}"
    cv2.putText(img,diemString,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.5,[0,255,0],3)
    Diem.append(diem)
    return


        # cv2.imshow('vc', VungChonedged)
        # cv2.imshow('vc1', VungChon0)
        # cv2.waitKey(50)


response_msv = []
def XuLyMsv():
# Xử lý mã sinh viên
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


    # Chia lưới cho MSV
    for col in range(6):
        min_white_pixels_msv = float('inf')
        selected_digit_msv = None
        selected_region_coords = None  # Thêm biến lưu tọa độ hình chữ nhật được chọn

        for row in range(10):
            # Vùng chọn cho MSV
            x_msv = msv_box[0] + col * (msv_box[2] // 6)
            y_msv = msv_box[1] + row * (msv_box[3] // 10)
            sub_region_msv = grey[y_msv:y_msv + (msv_box[3] // 10), x_msv:x_msv + (msv_box[2] // 6)]

            # Áp dụng bộ lọc bilateral để giữ chi tiết và làm mịn ảnh
            sub_region_msv_blur = cv2.bilateralFilter(sub_region_msv, d=9, sigmaColor=50, sigmaSpace=50)
            sub_region_msv_edged = cv2.Canny(sub_region_msv_blur, 75, 200)

            # Tìm các đường viền trong vùng con
            contours_msv, _ = cv2.findContours(sub_region_msv_edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_msv = np.zeros(sub_region_msv.shape, dtype="uint8")
            cv2.drawContours(mask_msv, contours_msv, -1, 255, -1)

            white_pixels_msv = cv2.countNonZero(mask_msv)
            # cv2.imshow('mask', mask_msv)
            # cv2.waitKey(50)
            # print(f"Col {col}, Row {row}, White pixels: {white_pixels_msv}")  # Debugging output

            # Điều kiện chọn ô có số lượng pixel trắng ít nhất
            if white_pixels_msv < min_white_pixels_msv:
                min_white_pixels_msv = white_pixels_msv
                selected_digit_msv = row
                selected_region_coords = (x_msv, y_msv, x_msv + (msv_box[2] // 6), y_msv + (msv_box[3] // 10))
        response_msv.append(selected_digit_msv)


        # Vẽ hình chữ nhật vào khu vực được chọn
        if selected_region_coords:
            cv2.rectangle(img, (selected_region_coords[0], selected_region_coords[1]),
                          (selected_region_coords[2], selected_region_coords[3]), (0, 255, 0), 2)


    result = ''.join(map(str, response_msv))
    cv2.putText(img,result,(msv_box[0]+2, msv_box[1]-15),1,1.5
                ,(0,255,0),2)
    return


response_idtest = []
def XuLyMaDe():
     #Xử lý mã đề
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



        # Chia lưới cho mã đề
        for col in range(3):
            min_white_pixels_idtest = float('inf')
            # min_white_pixels_idtest = 0
            selected_digit_idtest = None
            selected_region_coords_idtest = None

            for row in range(10):
                # Vùng chọn cho mã đề
                x_idtest = idtest_box[0] + col * (idtest_box[2] // 3)
                y_idtest = idtest_box[1] + row * (idtest_box[3] // 10)
                sub_region_idtest = grey[y_idtest:y_idtest + (idtest_box[3] // 10),
                                    x_idtest:x_idtest + (idtest_box[2] // 3)]

                # Áp dụng bộ lọc bilateral để giữ chi tiết và làm mịn ảnh
                sub_region_idtest_blur = cv2.bilateralFilter(sub_region_idtest, d=5, sigmaColor=50, sigmaSpace=50)
                sub_region_idtest_edged = cv2.Canny(sub_region_idtest_blur, 25, 150)

                # Tìm các đường viền trong vùng con
                contours_idtest, _ = cv2.findContours(sub_region_idtest_edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                mask_idtest = np.zeros(sub_region_idtest.shape, dtype="uint8")
                cv2.drawContours(mask_idtest, contours_idtest, -1, 255, -1)

                white_pixels_idtest = cv2.countNonZero(mask_idtest)
                # cv2.imshow('mask_idtest', mask_idtest)
                # cv2.waitKey(50)
                # print(f"Col {col}, Row {row}, White pixels: {white_pixels_idtest}")  # Debugging output

                # Điều kiện chọn ô có số lượng pixel trắng ít nhất
                if white_pixels_idtest < min_white_pixels_idtest:
                    min_white_pixels_idtest = white_pixels_idtest
                    selected_digit_idtest = row
                    selected_region_coords_idtest = (
                    x_idtest, y_idtest, x_idtest + (idtest_box[2] // 3), y_idtest + (idtest_box[3] // 10))

            response_idtest.append(selected_digit_idtest)

            # Vẽ hình chữ nhật vào khu vực được chọn cho mã đề
            if selected_region_coords_idtest:
                cv2.rectangle(img, (selected_region_coords_idtest[0], selected_region_coords_idtest[1]),
                              (selected_region_coords_idtest[2], selected_region_coords_idtest[3]), (0, 255, 0), 2)

    result1 = ''.join(map(str, response_idtest))
    idtest_box = boxIdtest[0]
    cv2.putText(img, result1, (idtest_box[0] + 2, idtest_box[1] - 15), 1, 1.5
                 , (0, 255, 0), 2)

    return


# Gọi hàm:
XuLyMsv()
XuLyMaDe()
print("\nMã sinh viên: ", response_msv)
print("Mã đề: ", response_idtest)
# Chuyển sang chuỗi
result = ''.join(map(str, response_msv))
result1 = ''.join(map(str, response_idtest))

print('mã sinh viên được tìm thấy trong ảnh: ' ,result)
print('mã đề được tìm thấy trong ảnh: ',result1)

# Kiểm tra mã đề có được tìm thấy không:
Check=False #Biến để kiểm tra có tìm được bài kiểm tra không

for baiKiemTra in list_bai_kiem_tra:
    print('Mã đề trong cơ sở dữ liệu',baiKiemTra.ma_de)
    if baiKiemTra.ma_de==result1:
        Check=True
        XulyDapAn(baiKiemTra)
        loaibaiktra=baiKiemTra.loai_baiKt
        break



if Check ==True:
    print("Mã đề đã được tìm thấy!!!!")
    print("Mã sinh viên: ", response_msv)
    print("Mã đề: ", response_idtest)
    if lbkt==1:print("Đây là mã đề cho bài kiểm tra số 1")
    elif lbkt==2:print("Đây là mã đề cho bài kiểm tra thứ 2")
    else: print("Đây là mã đề cho bài thi kết thúc học phần")
    print(result)
    print(result1)
    print(response)
    print("Số câu đúng",len(corectquest))
    print('Các câu đúng: ', corectquest)
    print('Điểm số: ',Diem)

else:
    print("Không tìm thấy mã đề")


# cv2.imshow('a', img)
# cv2.imshow('e', edged)
# mid = int(height * 0.3)
# botedged=edged[mid:height, :]
# botimg = img[mid:height, :]
# topimg=img[0:mid,:]
# cv2.imshow('botedged',botedged)
#
#
# cv2.imshow('botimg', botimg)
# cv2.imshow('topimg',topimg)
#
# cv2.waitKey()

from decimal import Decimal

for sv in listSV:
    print("mã sinh viên trong cơ sở dữ liệu: ", sv.ma_sinh_vien)
    if result == sv.ma_sinh_vien:
        print("Tìm thấy mã sinh viên ", result)
        print("Tên sinh viên được tìm thấy: ", sv.ten_sinh_vien)
        for diem in listdiem:
            if result == diem.ma_sinh_vien:
                # Kiểm tra nếu giá trị là None và thay thế bằng 0 hoặc giá trị phù hợp khác
                diem_kiem_tra_1 = diem.diem_kiem_tra_1 if diem.diem_kiem_tra_1 is not None else Decimal(0)
                diem_kiem_tra_2 = diem.diem_kiem_tra_2 if diem.diem_kiem_tra_2 is not None else Decimal(0)
                diem_thi_ket_thuc = diem.diem_thi_ket_thuc if diem.diem_thi_ket_thuc is not None else Decimal(0)

                if lbkt == 1:
                    diem_kiem_tra_1 = Decimal(Diem[0])
                elif lbkt == 2:
                    diem_kiem_tra_2 = Decimal(Diem[0])
                else:
                    diem_thi_ket_thuc = Decimal(Diem[0])

                diem1 = QuanLyDiem(
                    so_thu_tu=diem.so_thu_tu,
                    ma_sinh_vien=sv.ma_sinh_vien,
                    ten_sinh_vien=sv.ten_sinh_vien,
                    diem_10=diem.diem_10 if diem.diem_10 is not None else Decimal(0),
                    diem_kiem_tra_1=diem_kiem_tra_1,
                    diem_kiem_tra_2=diem_kiem_tra_2,
                    diem_thi_ket_thuc=diem_thi_ket_thuc
                )
                diem1.showInfor()

                # Chuyển đổi các giá trị Decimal sang float trước khi truyền vào câu lệnh UPDATE
                cursor.execute('''
                    UPDATE QuanLyDiem
                    SET diem_10_percent = ?,
                        diem_kiem_tra_1 = ?,
                        diem_kiem_tra_2 = ?,
                        diem_thi_ket_thuc = ?,
                        diem_40_percent = ?,
                        diem_trung_binh = ?
                    WHERE so_thu_tu = ? AND ma_sinh_vien = ?
                ''', (
                    float(diem1.diem_10),
                    float(diem1.diem_kiem_tra_1),
                    float(diem1.diem_kiem_tra_2),
                    float(diem1.diem_thi_ket_thuc),
                    float(diem1.diem_40),  # Giá trị điểm 40%
                    float(diem1.diem_trung_binh),  # Giá trị điểm trung bình
                    diem1.so_thu_tu,
                    diem1.ma_sinh_vien
                ))

                # Xác nhận cập nhật
                conn.commit()
                print("Đã cập nhật điểm vào cơ sở dữ liệu")

        timThaySV = True
        break

if not timThaySV:
    print("Không tìm thấy sinh viên")

conn.close()

mid = int(height * 0.3)
botedged=edged[mid:height, :]
botimg = img[mid:height, :]
topimg=img[0:mid,:]
cv2.imshow('botedged',botedged)


cv2.imshow('botimg', botimg)
cv2.imshow('topimg',topimg)

cv2.waitKey()
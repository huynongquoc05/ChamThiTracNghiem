import cv2

import numpy as np
import imutils

from imutils import contours
from imutils.perspective import four_point_transform
#Gọi tới 3 class Sinh vien ,bài kiểm tra và điểm lu trong file models
from Model import SinhVien,QuanLyDiem,BaiKiemTra
import pyodbc

class ChamDiem:
    def __init__(self):
        self.conn = pyodbc.connect(
            'Server=DESKTOP-AMN3OI8;Database=SinhVienVaDiem;UID=sa;PWD=051120;PORT=1433;DRIVER={SQL Server}')
        self.cursor = self.conn.cursor()

    def lay_du_lieu_sinh_vien(self):
        self.cursor.execute('select * from sinhvien')
        rows = self.cursor.fetchall()
        listSV = []
        for row in rows:
            sinhvien = SinhVien(ma_sinh_vien=row.ma_sinh_vien.strip(), ten_sinh_vien=row.ten_sinh_vien,
                                lop_hoc_phan=row.lop_hoc_phan)
            listSV.append(sinhvien)
        return listSV

    def lay_du_lieu_quan_ly_diem(self):
        self.cursor.execute('select * from quanlydiem')
        rows = self.cursor.fetchall()
        list_diem = []
        for row in rows:
            diem = QuanLyDiem(so_thu_tu=row.so_thu_tu, ma_sinh_vien=row.ma_sinh_vien.strip(),
                              ten_sinh_vien=row.ten_sinh_vien,
                              diem_10=row.diem_10_percent, diem_kiem_tra_1=row.diem_kiem_tra_1,
                              diem_kiem_tra_2=row.diem_kiem_tra_2, diem_thi_ket_thuc=row.diem_thi_ket_thuc)
            list_diem.append(diem)
        return list_diem

    def nhap_loai_bai_kiem_tra(self):
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

    def lay_du_lieu_bai_kiem_tra(self, loai_bai_kiem_tra):
        self.cursor.execute('select * from baikiemtra')
        rows = self.cursor.fetchall()
        list_bai_kiem_tra = []

        for row in rows:
            text = row.dap_an
            pairs = text.split(',')
            dictionary = {int(k): int(v) for k, v in (pair.split(':') for pair in pairs)}

            if row.loai_bai_kiem_tra == loai_bai_kiem_tra:
                bkt = BaiKiemTra(ma_de=row.ma_de.strip(), loai_baiKt=row.loai_bai_kiem_tra, so_cau_hoi=row.so_cau_hoi,
                                 dap_an=dictionary)
                list_bai_kiem_tra.append(bkt)

        return list_bai_kiem_tra

    def dong_ket_noi(self):
        self.conn.close()

    def doc_va_xu_ly_anh(self, image_path):
        # Đọc ảnh từ đường dẫn
        img = cv2.imread(image_path)

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blubled = cv2.GaussianBlur(grey, (3, 3), 0)
        edged = cv2.Canny(blubled, 0, 100)

        Vien = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Vien = imutils.grab_contours(Vien)
        Khung = []
        if len(Vien) > 0:

            Vien = sorted(Vien, key=cv2.contourArea, reverse=True)
            # docCnt=None
            for c in Vien:

                approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
                area = cv2.contourArea(c)
                if len(approx) == 4 and area > 20000:
                    # docCnt=approx
                    Khung.append(c)
        # print(len(Khung))

        # Lọc ra 2 cột mã sinh viên và mã đề
        msvAndIdtest = []
        for k in Khung:
            x, y, w, h = cv2.boundingRect(k)
            if h - 1.5 * w > 0 and h * w < 150000:
                msvAndIdtest.append(k)
        cv2.drawContours(img, msvAndIdtest, -1, (0, 255, 0), 3)
        # Cắt lấy vùng msv
        docCnt1 = approx = cv2.approxPolyDP(msvAndIdtest[0], 0.02 * cv2.arcLength(msvAndIdtest[0], True), True)
        paper1 = four_point_transform(img, docCnt1.reshape(4, 2))
        warped1 = four_point_transform(grey, docCnt1.reshape(4, 2))
        papermsv=[paper1,warped1]
        # Cắt lấy vùng id test
        docCnt2 = cv2.approxPolyDP(msvAndIdtest[1], 0.02 * cv2.arcLength(msvAndIdtest[1], True), True)
        paper2 = four_point_transform(img, docCnt2.reshape(4, 2))
        warped2 = four_point_transform(grey, docCnt2.reshape(4, 2))
        paperidt=[paper2,warped2]
        # Lọc ra 2 cột đáp án
        quesBox = []
        for k in Khung:

            x, y, w, h = cv2.boundingRect(k)
            if h - 1.5 * w > 0 and h * w > 150000:
                quesBox.append(k)

        quesBox = contours.sort_contours(quesBox, method="left-to-right")[0]
        # print("số phần tử trong questBox", len(quesBox))
        questCol=[]
        for quest in quesBox:
            docCnt = approx = cv2.approxPolyDP(quest, 0.02 * cv2.arcLength(quest, True), True)
            paper = four_point_transform(img, docCnt.reshape(4, 2))
            warped = four_point_transform(grey, docCnt.reshape(4, 2))
            col=[paper,warped]
            questCol.append(col)

        return questCol,papermsv,paperidt,img

    def XulyDapAn(self, object_bkt, dict_doc_vaXLA):
        response = {}
        corectquest = {}
        img_paper = []
        Diem = []
        DapAnMau = object_bkt.dap_an
        question_index = 1
        questCol=dict_doc_vaXLA[0]
        for col in questCol:
            paper,warped=col
            # cv2.imshow('paper',paper[500:,:])

            # print(paper.shape)
            h, w, c = paper.shape
            # x, y, w, h = cv2.boundingRect(paper)
            h_con = h // 5
            y_con = 0

            for i in range(5):

                # Xác định vùng hình chữ nhật con
                VungChon0 = paper[y_con + 15:y_con + h_con, :]
                VungChon = warped[y_con + 15:y_con + h_con, :]
                # Phóng to vùng chọn để dễ dàng xử lý
                scaleh = VungChon0.shape[0] * 3
                scalew = VungChon0.shape[1] * 3
                VungChonScale0 = cv2.resize(VungChon0, (scalew, scaleh))
                VungChonScale = cv2.resize(VungChon, (scalew, scaleh))
                VungChonblr = cv2.GaussianBlur(VungChonScale, (3, 3), 0)

                VungChonedged = cv2.Canny(VungChonblr, 75, 200)
                thresh = cv2.threshold(VungChonblr, 25, 200, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                # cv2.imshow('vccc',VungChonedged)

                vienOTracN = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # # Tìm đường viền của các ô trắc nghiệm trong mỗi hình chữ nhật con
                vienOTracN = imutils.grab_contours(vienOTracN)
                questionVien = []

                for v in vienOTracN:

                    peri = cv2.arcLength(v, True)
                    approx = cv2.approxPolyDP(v, 0.01 * peri, True)
                    if len(approx) > 6:

                        x1, y1, w1, h1 = cv2.boundingRect(v)
                        ar = w1 / float(h1)
                        area = cv2.contourArea(v)
                        if area < 50 and area > 10000:
                            continue  #

                        circularity = (4 * np.pi * area) / (cv2.arcLength(v, True) ** 2)

                        # Lọc các ô trắc nghiệm
                        if w1 >= 10 and h1 >= 10 and 0.9 <= ar <= 1.15:
                            questionVien.append(v)
                # print(len(questionVien))
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
                        # print('question', question_index, 'index', j, 'white pixel', total)
                        # Lấy ô ít pixel trắng nhất
                        if bubbled is None or total > bubbled[0]:
                            bubbled = (total, j)
                    color = (0, 0, 255)

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

                    cv2.drawContours(VungChonScale0, [vienOTracN[ans_j]], -1, color, 5)
                    # Thêm chỉ số câu hỏi và đáp án vào từ điển response
                    response[question_index] = ans_j_convert

                    question_index += 1
                # cv2.imshow('thress', thresh)
                restoreh = int(VungChonScale0.shape[0] / 3)
                restorew = int(VungChonScale0.shape[1] / 3)
                VungChonScale0 = cv2.resize(VungChonScale0, (restorew, restoreh))
                # cv2.imshow('vssc0', VungChonScale0)
                # cv2.waitKey(3000)
                paper[y_con + 15:y_con + h_con, :] = VungChonScale0
                y_con += h_con

            cv2.imwrite("NOT IN USE/maugiaycat.png", paper)
            cv2.imwrite('NOT IN USE/maugiaycat1.png', paper)
            # cv2.imshow('paper', paper)
            img_paper.append(paper)
        diem = 10 / object_bkt.so_cau_hoi * len(corectquest)
        Diem=diem
        diemString = f"Diem So: {diem:.2f}"
        cv2.putText(img_paper[1], diemString, (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.7, [255, 0, 0], 5)

        return img_paper,Diem,response,corectquest


    def XuLyMsv(self, dict_doc_vaXLA):
        paper1,warped1=dict_doc_vaXLA[1]
        response_msv = []
        x_curr = 0
        y_curr = 0
        h_msv, w_msv, c_msv = paper1.shape
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
                    # print(f"Col {col}, Row {row}, White pixels: {white_pixels_msv}")  # Debugging output
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
        result = ''.join(map(str, response_msv))

        return paper1,response_msv, result

    def XuLyMaDe(self, img_path):
        dict_doc_vaXLA =self.doc_va_xu_ly_anh(img_path)
        paper2,warped2=dict_doc_vaXLA[2]
        response_idt = []
        x_curr = 0
        y_curr = 0
        h_idt, w_idt, c_idt = paper2.shape
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

            if selected_region_coords:
                cv2.rectangle(paper2, (selected_region_coords[0], selected_region_coords[1]),
                              (selected_region_coords[2], selected_region_coords[3]), (0, 255, 0), 2)
        result1 = ''.join(map(str, response_idt))
        return paper2,response_idt, result1,dict_doc_vaXLA


    def layDuLieuChung(self, loai_bai_kiem_tra):
        # Gọi các phương thức để lấy dữ liệu
        list_sinh_vien = self.lay_du_lieu_sinh_vien()
        list_diem = self.lay_du_lieu_quan_ly_diem()
        list_bai_kiem_tra = self.lay_du_lieu_bai_kiem_tra(loai_bai_kiem_tra)

        # Trả về 3 danh sách
        return list_sinh_vien, list_diem, list_bai_kiem_tra

    def Cham(self, lbkt, img_path):
        # Lấy danh sách bài kiểm tra dựa trên loại bài kiểm tra
        list_bai_kiem_tra = self.layDuLieuChung(loai_bai_kiem_tra=lbkt)[2]

        # Kiểm tra nếu danh sách bài kiểm tra không rỗng
        if list_bai_kiem_tra:
            # Gọi phương thức xử lý mã đề
            img, response_idtest, result1,dict_doc_vaXLA = self.XuLyMaDe(img_path)

            # In kết quả mã đề và dữ liệu nhận diện mã đề
            print("Mã đề được nhận diện:", response_idtest)
            print("Kết quả mã đề:", result1)

            # Lặp qua danh sách bài kiểm tra và tìm bài kiểm tra với mã đề khớp
            for bkt in list_bai_kiem_tra:
                if result1 == bkt.ma_de:
                    print(f"=======Tìm thấy bài kiểm tra số {lbkt} với mã đề {result1}======= ")
                    # Gọi phương thức xử lý đáp án cho bài kiểm tra này
                    img, Diem, response, correctquest = self.XulyDapAn(bkt, dict_doc_vaXLA)

                    # In điểm và kết quả đáp án

                    print("\nCâu trả lời:", response)
                    print("\nCâu đúng:", correctquest)
                    print("\nSố câu đúng",len(correctquest))
                    print("Điểm:", Diem)

                    # Trả về điểm nếu mã đề khớp
                    return Diem,img,response, correctquest,dict_doc_vaXLA

            # Nếu không có bài kiểm tra nào khớp mã đề
            return "Bài kiểm tra không khớp mã đề."
        else:
            # Trả về chuỗi thông báo nếu không tìm thấy bài kiểm tra
            return "Bài kiểm tra không được tìm thấy."

    def capNhatCSDL(self, diem, lbkt):

        img, response_msv, result = self.XuLyMsv(diem[4])
        Diem=diem[0]
        print("Mã sinh viên được nhận diện:", response_msv)
        print("Kết quả m sinh viên:", result)

        list_sinh_vien, list_diem = self.layDuLieuChung(lbkt)[:2]
        from decimal import Decimal

        timThaySV = False  # Biến để kiểm tra có tìm thấy sinh viên hay không

        for sv in list_sinh_vien:
            print("mã sinh viên trong cơ sở dữ liệu: ", sv.ma_sinh_vien)
            if result == sv.ma_sinh_vien:
                CapNhat=False
                print("Tìm thấy mã sinh viên ", result)
                print("Tên sinh viên được tìm thấy: ", sv.ten_sinh_vien)

                for diem in list_diem:
                    if result == diem.ma_sinh_vien:
                        # Kiểm tra nếu giá trị là None và thay thế bằng 0 hoặc giá trị phù hợp khác
                        diem_kiem_tra_1 = diem.diem_kiem_tra_1 if diem.diem_kiem_tra_1 is not None else Decimal(0)
                        diem_kiem_tra_2 = diem.diem_kiem_tra_2 if diem.diem_kiem_tra_2 is not None else Decimal(0)
                        diem_thi_ket_thuc = diem.diem_thi_ket_thuc if diem.diem_thi_ket_thuc is not None else Decimal(0)

                        if lbkt == 1:
                            diem_kiem_tra_1 = Decimal(Diem)
                        elif lbkt == 2:
                            diem_kiem_tra_2 = Decimal(Diem)
                        else:
                            diem_thi_ket_thuc = Decimal(Diem)

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
                        self.cursor.execute('''
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
                        self.conn.commit()
                        print("Đã cập nhật điểm vào cơ sở dữ liệu")
                        CapNhat=True
                        break  # Thêm dòng này để thoát khỏi vòng lặp for
                if CapNhat==False:
                    print("Bảng điểm của sinh viên này chưa được tạo")
                    diem_10_percent = diem_kiem_tra_1 = diem_kiem_tra_2 = diem_thi_ket_thuc = 0
                    if lbkt == 1:
                        diem_kiem_tra_1 = Decimal(Diem)
                    elif lbkt == 2:
                        diem_kiem_tra_2 = Decimal(Diem)
                    else:
                        diem_thi_ket_thuc = Decimal(Diem)

                    diem1 = QuanLyDiem(
                        so_thu_tu=None,
                        ma_sinh_vien=sv.ma_sinh_vien,
                        ten_sinh_vien=sv.ten_sinh_vien,
                        diem_10=diem_10_percent,
                        diem_kiem_tra_1=diem_kiem_tra_1,
                        diem_kiem_tra_2=diem_kiem_tra_2,
                        diem_thi_ket_thuc=diem_thi_ket_thuc
                    )
                    self.cursor.execute('''
                            Insert into  QuanLyDiem
                            (ma_sinh_vien,ten_sinh_vien,diem_10_percent,diem_kiem_tra_1,
                            diem_kiem_tra_2,diem_40_percent,diem_thi_ket_thuc,diem_trung_binh) values
                            (?,?,?,?,?,?,?,?)
                        ''', (

                        diem1.ma_sinh_vien,
                        diem1.ten_sinh_vien,
                        float(diem1.diem_10),
                        float(diem1.diem_kiem_tra_1),
                        float(diem1.diem_kiem_tra_2),
                        float(diem1.diem_40),
                        float(diem1.diem_thi_ket_thuc),

                        float(diem1.diem_trung_binh)  # Giá trị điểm trung bình

                    ))
                    self.conn.commit()
                    print("Đã thêm bản ghi vào cơ sở dữ liệu")

                timThaySV = True
                break

        if timThaySV:
            return "Đã cập nhật điểm vào cơ sở dữ liệu"
        else:
            print('Không tìm thấy sinh viên')
            return "Không tìm thấy sinh viên"

if __name__ == '__main__':
    try:
        cham_diem = ChamDiem()
        lbkt = 1
        img_path = ('TEST_50C/AnhSo10.jpg')
        print(f'____________Đang tìm bài kiểm tra số {lbkt}__________')

        diem = cham_diem.Cham(lbkt, img_path=img_path)

        paper2, response_idtest, result1, dict_doc_vaXLA = cham_diem.XuLyMaDe(img_path)
        paper2 = cv2.resize(paper2, (66, 372))

        # Ảnh gốc
        img = dict_doc_vaXLA[3]

        paper1, response_msv, result = cham_diem.XuLyMsv(dict_doc_vaXLA)
        paper1 = cv2.resize(paper1, (149, 372))

        cv2.imshow(f'{img_path}', img)

        cv2.imshow('msv', paper1)
        cv2.imshow('idtest', paper2)
        # Kiểm tra kết quả trả về từ hàm Cham
        if isinstance(diem, str):
            print(diem)  # In ra thông báo nếu là chuỗi
        else:
            cham_diem.capNhatCSDL(diem, lbkt)

            # 2 cột đáp án
            img_paper = diem[1]
            for idx, img in enumerate(img_paper):
                img = cv2.resize(img, (320, 750))
                cv2.imshow(f'Image {idx + 1}', img)
            cv2.waitKey()

    except Exception as e:
        print("Đã xảy ra lỗi:", e)


    finally:
        # Đóng tất cả kết nối với CSDL hoặc các tài nguyên khác nếu có

        cham_diem.dong_ket_noi()
        print("Đã đóng kết nối")




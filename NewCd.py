
import time
starttime=time.time()
import cv2
import numpy as np
import imutils
import requests
from imutils import contours
from Model import SinhVien, QuanLyDiem, BaiKiemTra
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

        # Thay đổi kích thước ảnh
        img = cv2.resize(img, (786, 1118))

        # Lấy kích thước ảnh
        height, width = img.shape[:2]

        # Chuyển ảnh sang màu xám
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Làm mờ ảnh để giảm nhiễu
        blurred = cv2.GaussianBlur(grey, (3, 3), 0)

        # Phát hiện cạnh trong ảnh
        edged = cv2.Canny(blurred, 75, 200)

        # Trả về các ảnh đã xử lý
        return img, grey, blurred, edged, height, width

    def XulyDapAn(self, object_bkt,dict_DocvaXLA):  # Truyền đối tượng bài kiểm tra và đường dẫn ảnh
        response = {}
        corectquest = {}
        Diem = []
        # Gọi hàm đọc ảnh để lấy các ảnh đã xử lý và các thông số cần thiết
        img, grey, blurred, edged, height, width = dict_DocvaXLA
        # img=self.XuLyMsv(img_path)[0]

        # Lấy đáp án từ đối tượng
        DapAnMau = object_bkt.dap_an

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
        SoCauToiDa = object_bkt.so_cau_hoi
        corr = 0
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

                # Phóng to vùng chọn để dễ dàng xử lý
                scaleh = VungChon0.shape[0] * 3
                scalew = VungChon0.shape[1] * 3
                VungChonScale0 = cv2.resize(VungChon0, (scalew, scaleh))
                VungChonScale = cv2.resize(VungChon, (scalew, scaleh))
                VungChonblr = cv2.GaussianBlur(VungChonScale, (3, 3), 0)

                thresh = cv2.threshold(VungChonblr, 25, 200, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                # cv2.imshow('thresh',thresh)
                # cv2.waitKey()
                # Tìm đường viền của các ô trắc nghiệm trong mỗi hình chữ nhật con
                vienOTracN = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                vienOTracN = imutils.grab_contours(vienOTracN)
                questionVien = []

                for v in vienOTracN:
                    peri = cv2.arcLength(v, True)
                    approx = cv2.approxPolyDP(v, 0.01 * peri, True)
                    if len(approx) > 6:
                        x1, y1, w1, h1 = cv2.boundingRect(v)
                        ar = w1 / float(h1)
                        area = cv2.contourArea(v)
                        if area == 0 or area < 50:
                            continue  # Bỏ qua các đối tượng không có diện tích hoặc diện tích nhỏ
                        circularity = (4 * np.pi * area) / (cv2.arcLength(v, True) ** 2)

                        # Lọc các ô trắc nghiệm có dạng hình tròn với độ tròn > 0.7
                        if w1 >= 10 and h1 >= 10 and 0.85 <= ar <= 1.15 and circularity > 0.75:
                            questionVien.append(v)

                questionVien = contours.sort_contours(questionVien, method="top-to-bottom")[0]
                # print(len(questionVien))
                # Duyệt lấy câu trả lời (Đáp án)
                for (q, i) in enumerate(np.arange(0, len(questionVien), 4)):
                    vienOTracN = contours.sort_contours(questionVien[i:i + 4], method="left-to-right")[0]
                    bubbled = None
                    for (j, c) in enumerate(vienOTracN):
                        mask = np.zeros(thresh.shape, dtype='uint8')
                        cv2.drawContours(mask, [c], -1, 255, -1)
                        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                        total = cv2.countNonZero(mask)

                        # Lấy ô ít pixel trắng nhất
                        if bubbled is None or total > bubbled[0]:
                            bubbled = (total, j)

                    color = (0, 0, 255)
                    k = DapAnMau[question_index]  # lấy đáp án mẫu của câu hỏi
                    ans_j = bubbled[1]  # Lấy chỉ số ô được chọn (đáp án)
                    ans_j_convert = ['A', 'B', 'C', 'D'][ans_j]

                    if k == ans_j:
                        color = (0, 255, 0)
                        corectquest[question_index] = ans_j_convert

                    cv2.drawContours(VungChonScale0, [vienOTracN[ans_j]], -1, color, 5)
                    response[question_index] = ans_j_convert

                    question_index += 1
                    if question_index > SoCauToiDa:
                        break
                restoreh = int(VungChonScale0.shape[0] / 3)
                restorew = int(VungChonScale0.shape[1] / 3)
                VungChonScale0 = cv2.resize(VungChonScale0, (restorew, restoreh))
                # cv2.imshow('vssc0', VungChonScale0)
                # cv2.waitKey(500)
                img[y_con:y_con + h_con, x:x + w] = VungChonScale0
                y_con += h_con

        diem = 10 / object_bkt.so_cau_hoi * len(corectquest)
        diemString = f"Diem so: {diem:.2f}"
        cv2.putText(img, diemString, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, [0, 255, 0], 4)
        Diem=diem

        return img, Diem, response, corectquest

    def XuLyMsv(self, dict_DocvaXLA):

        response_msv = []
        img = dict_DocvaXLA[0]
        grey = dict_DocvaXLA[1]
        edged = dict_DocvaXLA[3]
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
        cv2.drawContours(img, [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]) for (x, y, w, h) in boxMsv],
                         -1,
                         (255, 0, 0), 1)

        # Xử lý mã sinh viên
        msv_box = boxMsv[0]

        # Chia lưới cho MSV
        for col in range(6):
            max_white_pixels_msv = 0
            selected_digit_msv = None
            selected_region_coords = None  # Lưu tọa độ hình chữ nhật được chọn

            for row in range(10):
                # Vùng chọn cho MSV
                x_msv = msv_box[0] + col * (msv_box[2] // 6)
                y_msv = msv_box[1] + row * (msv_box[3] // 10)
                sub_region_msv = grey[y_msv + 2:y_msv + (msv_box[3] // 10) - 2, x_msv + 2:x_msv + (msv_box[2] // 6) - 2]

                # Áp dụng bộ lọc bilateral để giữ chi tiết và làm mịn ảnh
                sub_region_msv_blur = cv2.bilateralFilter(sub_region_msv, d=9, sigmaColor=50, sigmaSpace=50)
                thresh = cv2.threshold(sub_region_msv_blur, 25, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                # cv2.imshow('th', thresh)
                # cv2.waitKey()

                white_pixels_msv = cv2.countNonZero(thresh)

                # print(f"Col {col}, Row {row}, White pixels: {white_pixels_msv}")  # Debugging output

                # Điều kiện chọn ô có số lượng pixel trắng lớn nhất
                if white_pixels_msv > max_white_pixels_msv:
                    max_white_pixels_msv = white_pixels_msv
                    selected_digit_msv = row
                    selected_region_coords = (x_msv, y_msv, x_msv + (msv_box[2] // 6), y_msv + (msv_box[3] // 10))

            # Lưu số hàng (digit) đã chọn cho cột MSV này
            response_msv.append(selected_digit_msv)

            # Vẽ hình chữ nhật vào khu vực được chọn
            if selected_region_coords:
                cv2.rectangle(img, (selected_region_coords[0], selected_region_coords[1]),
                              (selected_region_coords[2], selected_region_coords[3]), (0, 255, 0), 2)

        result = ''.join(map(str, response_msv))
        cv2.putText(img, result, (msv_box[0] + 2, msv_box[1] - 15), 1, 1.5

                    , (0, 255, 0), 2)
        dict_DocvaXLA[0]=img
        return dict_DocvaXLA, response_msv, result

    def XuLyMaDe(self, img_path):
        response_idtest = []
        dict_DocvaXLA=self.doc_va_xu_ly_anh(img_path)
        img = dict_DocvaXLA[0]
        grey = dict_DocvaXLA[1]
        edged = dict_DocvaXLA[3]
        # Xử lý mã đề
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
        cv2.drawContours(img,
                         [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]) for (x, y, w, h) in boxIdtest], -1,
                         (255, 0, 0), 1)

        # Xử lý mã đề
        if len(boxIdtest) > 0:
            idtest_box = boxIdtest[0]  # Giả định rằng vùng đầu tiên là mã đề

            # Chia lưới cho mã đề
            for col in range(3):
                max_white_pixels_idtest = 0
                # min_white_pixels_idtest = 0
                selected_digit_idtest = None
                selected_region_coords_idtest = None

                for row in range(10):
                    # Vùng chọn cho mã đề
                    x_idtest = idtest_box[0] + col * (idtest_box[2] // 3)
                    y_idtest = idtest_box[1] + row * (idtest_box[3] // 10)
                    sub_region_idtest = grey[y_idtest + 2:y_idtest + (idtest_box[3] // 10) - 2,
                                        x_idtest + 2:x_idtest - 2 + (idtest_box[2] // 3)]

                    # Áp dụng bộ lọc bilateral để giữ chi tiết và làm mịn ảnh
                    sub_region_idtest_blur = cv2.bilateralFilter(sub_region_idtest, d=7, sigmaColor=50, sigmaSpace=50)

                    thresh = cv2.threshold(sub_region_idtest_blur, 25, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    # cv2.imshow('th', thresh)
                    # cv2.waitKey()

                    white_pixels_idtest = cv2.countNonZero(thresh)
                    # cv2.imshow('mask_idtest', mask_idtest)
                    # cv2.waitKey(300)
                    # print(f"Col {col}, Row {row}, White pixels: {white_pixels_idtest}")  # Debugging output

                    # Điều kiện chọn ô có số lượng pixel trắng nhiều nhất nhất
                    if white_pixels_idtest > max_white_pixels_idtest:
                        max_white_pixels_idtest = white_pixels_idtest
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

        return dict_DocvaXLA, response_idtest, result1,img

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
            dict_DocvaXLA, response_idtest, result1, afterimg = self.XuLyMaDe(img_path)
            # cv2.imshow('aaa',dict_DocvaXLA[0])
            # cv2.waitKey()
            # In kết quả mã đề và dữ liệu nhận diện mã đề
            print("Mã đề được nhận diện:", response_idtest)
            print("Kết quả mã đề:", result1)

            # Lặp qua danh sách bài kiểm tra và tìm bài kiểm tra với mã đề khớp
            for bkt in list_bai_kiem_tra:
                if result1 == bkt.ma_de:
                    print(f'======Tìm thấy bài kiểm tra số {lbkt} với mã đề {result1}======')
                    # Gọi phương thức xử lý đáp án cho bài kiểm tra này
                    img, Diem, response, correctquest = self.XulyDapAn(bkt, dict_DocvaXLA)

                    # In điểm và kết quả đáp án
                    print("\nĐiểm:", Diem)
                    print("Câu trả lời: ", response)
                    print("\nCâu đúng: ", correctquest)
                    print(f"Số câu đúng: {len(correctquest)}\n",)
                    # Trả về điểm nếu mã đề khớp
                    dict_DocvaXLA = list(dict_DocvaXLA)  # Chuyển đổi tuple thành list
                    # dict_DocvaXLA[0] = img  # Thực hiện thao tác gán

                    return Diem, img,dict_DocvaXLA

            # Nếu không có bài kiểm tra nào khớp mã đề
            return "Bài kiểm tra không khớp mã đề."
        else:
            # Trả về chuỗi thông báo nếu không tìm thấy bài kiểm tra
            return "Bài kiểm tra không được tìm thấy."

    def capNhatCSDL(self, diem_tupleOBJ, lbkt):
        img, response_msv, result = self.XuLyMsv(diem_tupleOBJ[2])


        print("Mã sinh viên được nhận diện:", response_msv)
        print("Kết quả m sinh viên:", result)

        list_sinh_vien, list_diem = self.layDuLieuChung(lbkt)[:2]
        from decimal import Decimal

        timThaySV = False  # Biến để kiểm tra có tìm thấy sinh viên hay không

        for sv in list_sinh_vien:
            print("mã sinh viên trong cơ sở dữ liệu: ", sv.ma_sinh_vien)
            if result == sv.ma_sinh_vien:
                print("Tìm thấy mã sinh viên ", result)
                print("Tên sinh viên được tìm thấy: ", sv.ten_sinh_vien)
                CapNhat = False
                for diem in list_diem:
                    if result == diem.ma_sinh_vien:
                        # Kiểm tra nếu giá trị là None và thay thế bằng 0 hoặc giá trị phù hợp khác
                        diem_kiem_tra_1 = diem.diem_kiem_tra_1 if diem.diem_kiem_tra_1 is not None else Decimal(0)
                        diem_kiem_tra_2 = diem.diem_kiem_tra_2 if diem.diem_kiem_tra_2 is not None else Decimal(0)
                        diem_thi_ket_thuc = diem.diem_thi_ket_thuc if diem.diem_thi_ket_thuc is not None else Decimal(0)

                        if lbkt == 1:
                            diem_kiem_tra_1 = Decimal(diem_tupleOBJ[0])
                        elif lbkt == 2:
                            diem_kiem_tra_2 = Decimal(diem_tupleOBJ[0])
                        else:
                            diem_thi_ket_thuc = Decimal(diem_tupleOBJ[0])

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
                        CapNhat = True
                        break  # Thêm dòng này để thoát khỏi vòng lặp for
                if CapNhat == False:
                    print("Bảng điểm của sinh viên này chưa được tạo")
                    diem_10_percent = diem_kiem_tra_1 = diem_kiem_tra_2 = diem_thi_ket_thuc = 0
                    if lbkt == 1:
                        diem_kiem_tra_1 = Decimal(diem_tupleOBJ[0])
                    elif lbkt == 2:
                        diem_kiem_tra_2 = Decimal(diem_tupleOBJ[0])
                    else:
                        diem_thi_ket_thuc = Decimal(diem_tupleOBJ[0])

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
            return "Đã cập nhật"
        else:
            return "Không tìm thấy sinh viên"


# Sử dụng các phương thức từ lớp ChamDiem
if __name__ == "__main__":
    cham_diem = ChamDiem()
    lbkt = 2
    img_path = 'Test/MauGiay1.png'
    diem = cham_diem.Cham(lbkt, img_path)
    # cv2.imshow('img0', diem[1])
    # print(cham_diem.Cham(lbkt,'MauGiay4.png'))
    cham_diem.capNhatCSDL(diem, lbkt)
    endtime=time.time()
    print("Thời gian chạy : ",endtime-starttime)
    cv2.imshow('img', diem[1])
    cv2.waitKey()
    # print(cham_diem.capNhatCSDL(diem,'MauGiay4.png',lbkt))


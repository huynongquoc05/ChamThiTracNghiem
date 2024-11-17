# Nơi đây chứa 3 class bài kiểm tra , sinh viên và điểm.

#Class Giảng viên
#Class Sinh Viên:
class SinhVien:
    def __init__(self, ma_sinh_vien, ten_sinh_vien, lop_hoc_phan):
        self.ma_sinh_vien = ma_sinh_vien
        self.ten_sinh_vien = ten_sinh_vien
        self.lop_hoc_phan = lop_hoc_phan

    def showInfor(self):
        print(
            f'Mã sinh viên: {self.ma_sinh_vien}, \nTên sinh viên: {self.ten_sinh_vien}, '
            f'\nLớp học phần: {self.lop_hoc_phan}')


#class Quản lý điểm
from decimal import Decimal

class QuanLyDiem:
    def __init__(self, so_thu_tu, ma_sinh_vien, ten_sinh_vien, diem_10, diem_kiem_tra_1, diem_kiem_tra_2, diem_thi_ket_thuc):
        self.so_thu_tu = so_thu_tu
        self.ma_sinh_vien = ma_sinh_vien
        self.ten_sinh_vien = ten_sinh_vien
        self.diem_10 = Decimal(diem_10) if diem_10 is not None else Decimal(0)
        self.diem_kiem_tra_1 = Decimal(diem_kiem_tra_1) if diem_kiem_tra_1 is not None else Decimal(0)
        self.diem_kiem_tra_2 = Decimal(diem_kiem_tra_2) if diem_kiem_tra_2 is not None else Decimal(0)
        self.diem_thi_ket_thuc = Decimal(diem_thi_ket_thuc) if diem_thi_ket_thuc is not None else Decimal(0)
        self.diem_40 = self.tinh_diem_40()
        self.diem_trung_binh = self.tinh_diem_trung_binh()

    def tinh_diem_40(self):
        return self.diem_kiem_tra_1 * Decimal(0.5) + self.diem_kiem_tra_2 * Decimal(0.5)

    def tinh_diem_trung_binh(self):
        return (
            self.diem_10 * Decimal(0.1) +
            self.diem_40 * Decimal(0.4) +
            self.diem_thi_ket_thuc * Decimal(0.5)
        )

    def showInfor(self):
        print(f'Số thứ tự: {self.so_thu_tu}')
        print(f'Mã sinh viên: {self.ma_sinh_vien}')
        print(f'Tên sinh viên: {self.ten_sinh_vien}')
        print(f'Điểm 10%: {self.diem_10:.2f}')
        print(f'Điểm KT1: {self.diem_kiem_tra_1:.2f}')
        print(f'Điểm KT2: {self.diem_kiem_tra_2:.2f}')
        print(f'Điểm 40%: {self.diem_40:.2f}')
        print(f'Điểm Kết thúc: {self.diem_thi_ket_thuc:.2f}')
        print(f'Điểm trung bình: {self.diem_trung_binh:.2f}')

# Class ba kiểm tra
class BaiKiemTra:
    def __init__(self, ma_de,loai_baiKt, so_cau_hoi, dap_an):
        self.ma_de = ma_de
        # Mã đề thi
        self.loai_baiKt=loai_baiKt
        self.so_cau_hoi = so_cau_hoi  # Số lượng câu hỏi
        self.dap_an = dap_an          # Từ điển đáp án, key là số câu hỏi, value là đáp án

    def showInfor(self):
        print(f'Mã đề: {self.ma_de}, \nLoại bài kiểm tra: {self.loai_baiKt}, \nSố câu hỏi: {self.so_cau_hoi},'
              f' \nĐáp án: {self.dap_an}')

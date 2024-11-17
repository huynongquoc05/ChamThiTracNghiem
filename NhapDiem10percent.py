from Model import QuanLyDiem


# Hàm nhập điểm 10%
def nhapdiem10pc():
    while True:
        try:
            diem10pc = float(input("Nhập điểm 10%: "))
            if 0 <= diem10pc <= 10:
                return diem10pc
            else:
                print("Giá trị không hợp lệ, vui lòng nhập lại.")
        except ValueError:
            print("Giá trị không hợp lệ, vui lòng nhập lại.")


# Kết nối cơ sở dữ liệu
def connect_db():
    import pyodbc
    conn = pyodbc.connect(
        'Server=DESKTOP-AMN3OI8;Database=SinhVienVaDiem;UID=sa;PWD=051120;PORT=1433;DRIVER={SQL Server}')
    return conn


# Hàm tìm đối tượng điểm
def timDiem(conn, stt, msv):
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM quanlydiem
        WHERE so_thu_tu = ? AND ma_sinh_vien = ?
    ''', (stt, msv))
    row = cursor.fetchone()
    if row:
        return QuanLyDiem(so_thu_tu=row.so_thu_tu, ma_sinh_vien=row.ma_sinh_vien.strip(),
                          ten_sinh_vien=row.ten_sinh_vien,
                          diem_10=row.diem_10_percent, diem_kiem_tra_1=row.diem_kiem_tra_1,
                          diem_kiem_tra_2=row.diem_kiem_tra_2, diem_thi_ket_thuc=row.diem_thi_ket_thuc)
    return None


# Hàm nhập điểm và cập nhật vào cơ sở dữ liệu
def nhapDiem():
    stt = int(input("Nhập số thứ tự: "))
    msv = input("Nhập mã sinh viên: ")

    try:
        conn = connect_db()
        timThaydiem = timDiem(conn, stt, msv)

        if timThaydiem is None:
            print("Không tìm thấy điểm cho số thứ tự", stt, 'và mã sinh viên', msv)
            choice = input("Bấm 1 để nhập lại STT và MSV, phím khác để kết thúc: ")
            if choice == '1':
                nhapDiem()  # Gọi lại hàm để người dùng nhập lại thông tin
            else:
                print("Kết thúc chương trình.")
                return
        else:
            print("Đối tượng điểm được tìm thấy: ")
            timThaydiem.showInfor()

            diem10pc = nhapdiem10pc()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE QuanLyDiem
                SET diem_10_percent = ?
                WHERE so_thu_tu = ? AND ma_sinh_vien = ?
            ''', (diem10pc, timThaydiem.so_thu_tu, timThaydiem.ma_sinh_vien))
            conn.commit()
            print("Đã cập nhật điểm vào cơ sở dữ liệu")
    except Exception as e:
        print("Đã xảy ra lỗi:", e)
    finally:
        conn.close()
        print("Đã đóng kết nối")


if __name__ == '__main__':
    nhapDiem()

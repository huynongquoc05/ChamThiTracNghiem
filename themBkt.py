from Model import BaiKiemTra


# Hàm nhập đáp án
def nhapDapAn(so_cau_hoi):
    dap_an_dict = {}
    for i in range(1, so_cau_hoi + 1):
        while True:
            try:
                dap_an = int(input(f"Nhập đáp án cho câu hỏi {i} (0-3): "))
                if dap_an in [0, 1, 2, 3]:
                    dap_an_dict[i] = dap_an
                    break
                else:
                    print("Đáp án không hợp lệ, vui lòng nhập lại.")
            except ValueError:
                print("Giá trị không hợp lệ, vui lòng nhập lại.")

    # Chuyển từ dictionary sang dạng text "1:0,2:1,3:2,..."
    dap_an_text = ','.join([f"{k}:{v}" for k, v in dap_an_dict.items()])
    return dap_an_text


# Kết nối cơ sở dữ liệu
def connect_db():
    import pyodbc
    conn = pyodbc.connect(
        'Server=DESKTOP-AMN3OI8;Database=SinhVienVaDiem;UID=sa;PWD=051120;PORT=1433;DRIVER={SQL Server}')
    return conn


# Hàm tìm đối tượng bài kiểm tra
def timBaiKiemTra(conn, ma_de, loai_bai_kiem_tra):
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM BaiKiemTra
        WHERE ma_de = ? AND loai_bai_kiem_tra = ?
    ''', (ma_de, loai_bai_kiem_tra))
    row = cursor.fetchone()
    if row:
        text = row.dap_an
        pairs = text.split(',')
        dictionary = {int(k): int(v) for k, v in (pair.split(':') for pair in pairs)}
        return BaiKiemTra(ma_de=row.ma_de.strip(), loai_baiKt=row.loai_bai_kiem_tra, so_cau_hoi=row.so_cau_hoi,
                          dap_an=dictionary)
    return None


def nhap_loai_bai_kiem_tra():
    while True:
        try:
            loai_bai_kiem_tra = int(input("Nhập loại bài kiểm tra: "))  # Không giới hạn giá trị
            return loai_bai_kiem_tra  # Trả về giá trị người dùng nhập
        except ValueError:
            print("Giá trị không hợp lệ, vui lòng nhập lại.")


def nhapSoCauHoi():
    while True:
        try:
            so_cau_hoi = int(input("Nhập số câu hỏi (<120): "))
            if 0 < so_cau_hoi <= 120:
                return so_cau_hoi
            else:
                print("Giá trị không hợp lệ, vui lòng nhập lại.")
        except ValueError:
            print("Giá trị không hợp lệ, vui lòng nhập lại.")


# Hàm nhập bài kiểm tra và cập nhật vào cơ sở dữ liệu
def nhapBaiKiemTra():
    ma_de = input("Nhập mã đề: ")
    loai_baiKt = nhap_loai_bai_kiem_tra()

    try:
        conn = connect_db()
        timThayBaiKT = timBaiKiemTra(conn, ma_de, loai_baiKt)

        if timThayBaiKT:
            print(f"Bài kiểm tra với mã đề {ma_de} và bài kiểm tra số {loai_baiKt} đã tồn tại.")
            choice = input("Bấm 1 để nhập lại mã đề và bài kiểm tra, phím khác để kết thúc: ")
            if choice == '1':
                nhapBaiKiemTra()  # Gọi lại hàm để người dùng nhập lại thông tin
            else:
                print("Kết thúc chương trình.")
                return
        else:
            print(f"Bài kiểm tra với mã đề {ma_de} và bài kiểm tra số {loai_baiKt} chưa được khởi tạo.")

            so_cau_hoi = nhapSoCauHoi()
            dap_an_text = nhapDapAn(so_cau_hoi)

            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO BaiKiemTra (ma_de, loai_bai_kiem_tra, so_cau_hoi, dap_an)
                VALUES (?, ?, ?, ?)
            ''', (ma_de, loai_baiKt, so_cau_hoi, dap_an_text))
            conn.commit()
            print("Đã thêm bài kiểm tra vào cơ sở dữ liệu")
    except Exception as e:
        print("Đã xảy ra lỗi:", e)
    finally:
        conn.close()
        print("Đã đóng kết nối")


if __name__ == '__main__':
    nhapBaiKiemTra()

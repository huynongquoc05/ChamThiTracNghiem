from contextlib import redirect_stdout

from Model import BaiKiemTra


# Hàm nhập đáp án
def ConvertDA(dap_an_dict):
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

import io
import io
from contextlib import redirect_stdout

def nhapBaiKiemTra(ma_de, loai_baiKt, so_cau_hoi, dictDA):
    output = io.StringIO()  # Tạo output ở ngoài để truy xuất dễ dàng

    with redirect_stdout(output):  # Đặt redirect_stdout ở đây để bắt tất cả các lệnh print
        try:
            conn = connect_db()
            timThayBaiKT = timBaiKiemTra(conn, ma_de, loai_baiKt)

            if timThayBaiKT:
                print(f"Bài kiểm tra với mã đề {ma_de} và bài kiểm tra số {loai_baiKt} đã tồn tại.")
                status = 'Failed to add BKT'
            else:
                print(f"Bài kiểm tra với mã đề {ma_de} và bài kiểm tra số {loai_baiKt} chưa được khởi tạo.")
                dap_an_text = ConvertDA(dictDA)

                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO BaiKiemTra (ma_de, loai_bai_kiem_tra, so_cau_hoi, dap_an)
                    VALUES (?, ?, ?, ?)
                ''', (ma_de, loai_baiKt, so_cau_hoi, dap_an_text))
                conn.commit()
                print("Đã thêm bài kiểm tra vào cơ sở dữ liệu")
                status = 'Successfully added BKT'
        except Exception as e:
            print("Đã xảy ra lỗi exception:", e)
            status = 'Failed to add BKT'
        finally:
            conn.close()
            print("Đã đóng kết nối")

    console_output = output.getvalue().replace("\n", "<br>")
    return status, console_output


if __name__ == '__main__':
    ma_de='123'
    loai_baiKt=3
    dictA = {1: 'C', 2: 'A', 3: 'C', 4: 'A', 5: 'C', 6: 'B', 7: 'D', 8: 'A', 9: 'C', 10: 'B', 11: 'A', 12: 'B', 13: 'C',
             14: 'A', 15: 'B', 16: 'C', 17: 'A', 18: 'B', 19: 'C', 20: 'B', 21: 'D', 22: 'A', 23: 'B', 24: 'C', 25: 'A',
             26: 'B', 27: 'A', 28: 'C', 29: 'A', 30: 'C', 31: 'C', 32: 'B', 33: 'C', 34: 'B', 35: 'A', 36: 'B', 37: 'C',
             38: 'A', 39: 'D', 40: 'A', 41: 'C', 42: 'A', 43: 'C', 44: 'B', 45: 'C', 46: 'A', 47: 'C', 48: 'B', 49: 'A',
             50: 'B', 51: 'C', 52: 'A', 53: 'B', 54: 'C', 55: 'B', 56: 'A', 57: 'C', 58: 'B', 59: 'B', 60: 'D', 61: 'A',
             62: 'A', 63: 'A', 64: 'A', 65: 'A', 66: 'A', 67: 'A', 68: 'A', 69: 'A', 70: 'A', 71: 'B', 72: 'B', 73: 'B',
             74: 'B', 75: 'B', 76: 'B', 77: 'B', 78: 'B', 79: 'B', 80: 'B', 81: 'B', 82: 'B', 83: 'B', 84: 'B', 85: 'B',
             86: 'B', 87: 'B', 88: 'B', 89: 'B', 90: 'B', 91: 'C', 92: 'C', 93: 'C', 94: 'C', 95: 'C', 96: 'C', 97: 'C',
             98: 'C', 99: 'C', 100: 'C', 101: 'C', 102: 'C', 103: 'C', 104: 'C', 105: 'C', 106: 'D', 107: 'D', 108: 'D',
             109: 'D', 110: 'D', 111: 'D', 112: 'D', 113: 'D', 114: 'D', 115: 'D', 116: 'D', 117: 'B', 118: 'B',
             119: 'B', 120: 'B'}

    KetQua=nhapBaiKiemTra(ma_de,loai_baiKt,120,dictA)
    print(f'Kết quả {KetQua}')

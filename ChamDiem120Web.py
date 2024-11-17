import traceback
import cv2
import numpy as np
from NewCd import ChamDiem
import io
from contextlib import redirect_stdout
import os

# Đường dẫn để lưu các ảnh đã xử lý tạm thời
SAVE_DIR = "static/processed_images"


# Hàm lưu ảnh và trả về đường dẫn ảnh
def save_image(image, filename):
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Lưu ảnh với đường dẫn cụ thể
    file_path = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(file_path, image)
    return file_path


def cham_bai_kiem_tra_120(lbkt, img_path):
    try:
        cham_diem = ChamDiem()

        # Bắt đầu bắt output console
        output = io.StringIO()
        with redirect_stdout(output):
            Made=cham_diem.XuLyMaDe(img_path)
            diem = cham_diem.Cham(lbkt, img_path)
            # cv2.imshow('img0', diem[1])
            # print(cham_diem.Cham(lbkt,'MauGiay4.png'))
            cham_diem.capNhatCSDL(diem, lbkt)
            if isinstance(diem, str):
                print(diem)
                made_path = save_image(Made[3], "made.png")
                console_output = output.getvalue().replace("\n", "<br>")
                return lbkt,made_path, diem, console_output
            else:
                ketqua_path=save_image(diem[1],'Ketqua.png')
                console_output = output.getvalue().replace("\n", "<br>")
                return lbkt,diem[0],ketqua_path,console_output

    except Exception as e:
        print("Đã xảy ra lỗi:", e)
        print("Chi tiết lỗi:")
        traceback.print_exc()
        return None
    finally:
        cham_diem.dong_ket_noi()
        print("Đã đóng kết nối")
        print(output.getvalue())




if __name__ == '__main__':
    cham_bai_kiem_tra_120(2, 'Test/MauGiay3.png')

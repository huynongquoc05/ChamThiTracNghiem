import traceback
import cv2
import numpy as np
from NewCd2 import ChamDiem
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


def cham_bai_kiem_tra(lbkt, img_path):
    try:
        print(f"Đang xử lý chấm ảnh {img_path}")
        # Tạo đối tượng ChamDiem
        cham_diem = ChamDiem()
        print(f'____________Đang tìm bài kiểm tra số {lbkt}__________')

        # Bắt đầu bắt output console
        output = io.StringIO()
        with redirect_stdout(output):
            # Mã sinh viên và mã đề
            paper2, response_idtest, result1, dict_doc_vaXLA = cham_diem.XuLyMaDe(img_path)
            paper2 = cv2.resize(paper2, (66, 372))

            # Ảnh gốc
            img = dict_doc_vaXLA[3]

            paper1, response_msv, result = cham_diem.XuLyMsv(dict_doc_vaXLA)
            paper1 = cv2.resize(paper1, (149, 372))

            # Lưu các ảnh đã xử lý vào đường dẫn tạm thời
            img_path_saved = save_image(img, "original_image.jpg")
            paper1_path = save_image(paper1, "paper1.jpg")
            paper2_path = save_image(paper2, "paper2.jpg")

            # Chấm bài kiểm tra và lấy điểm
            diem = cham_diem.Cham(lbkt, img_path=img_path)

            # Kiểm tra kết quả trả về từ hàm Cham
            if isinstance(diem, str):
                print(diem)
                console_output = output.getvalue().replace("\n", "<br>")
                return lbkt, img_path_saved, paper1_path, result, paper2_path, result1, diem, console_output
            else:
                # Nếu diem không phải là chuỗi thông báo
                confirm = cham_diem.capNhatCSDL(diem, lbkt)

                # Lưu ảnh từ danh sách img_paper
                img_paper_paths = [save_image(image, f"img_paper_{i}.jpg") for i, image in enumerate(diem[1])]
                response = diem[2]
                correctquest = diem[3]

                # Chuyển đổi output console
                console_output = output.getvalue().replace("\n", "<br>")
                return lbkt, img_path_saved, paper1_path, result, paper2_path, result1, diem[
                    0], img_paper_paths, response, correctquest, len(correctquest), confirm, console_output

    except Exception as e:
        print("Đã xảy ra lỗi:", e)
        print("Chi tiết lỗi:")
        traceback.print_exc()
        return None
    finally:
        cham_diem.dong_ket_noi()
        print("Đã đóng kết nối")




if __name__ == '__main__':
    cham_bai_kiem_tra(1, 'static/uploads/AnhSo10.jpg')

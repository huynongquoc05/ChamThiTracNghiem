import os
from NewCd import ChamDiem

chamdiem = ChamDiem()

lbkt = 2  # Loại bài kiểm tra

# Thư mục chứa các ảnh cần xử lý
thu_muc_anh = 'Test'

try:
    # Duyệt qua từng ảnh trong thư mục
    for ten_anh in os.listdir(thu_muc_anh):
        # Tạo đường dẫn đầy đủ cho ảnh
        img_path = os.path.join(thu_muc_anh, ten_anh)

        # Kiểm tra nếu tệp không phải là ảnh, tiếp tục sang tệp khác
        if not ten_anh.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"'{ten_anh}' không phải là định dạng ảnh, bỏ qua.")
            continue

        try:
            # Gọi hàm Cham để chấm điểm ảnh hiện tại
            print('\n Đang xử lý ảnh', img_path)
            diem = chamdiem.Cham(lbkt, img_path)
            print(f"Điểm của ảnh {ten_anh}: {diem[0]}")

            # Nếu mã đề không khớp hoặc không tìm thấy bài kiểm tra, bỏ qua ảnh này
            if diem is None:
                print(f"Mã đề của ảnh {ten_anh} không khớp hoặc không tìm thấy bài kiểm tra, bỏ qua ảnh.")
                continue

            # Cập nhật CSDL với điểm đã chấm
            ket_qua_cap_nhat = chamdiem.capNhatCSDL(diem, lbkt)
            print(f"Kết quả cập nhật ảnh {ten_anh}: {ket_qua_cap_nhat}")

        except Exception as e:
            # Nếu có lỗi trong quá trình xử lý ảnh, bỏ qua ảnh đó và xử lý tiếp
            print(f"Đã xảy ra lỗi khi xử lý ảnh {ten_anh}: {e}")
            continue

finally:
    # Đảm bảo phương thức đóng kết nối luôn được gọi
    print("\n Đang đóng kết nối...")
    chamdiem.dong_ket_noi()
    print("Kết nối đã được đóng.")
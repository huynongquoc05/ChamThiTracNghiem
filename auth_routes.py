from flask import Blueprint, render_template, request, redirect, url_for,flash,current_app
from ChamDiem50Web import cham_bai_kiem_tra
import os
from ChamDiem120Web import cham_bai_kiem_tra_120
auth = Blueprint('auth', __name__)


@auth.route('/Chamdiem50', methods=['GET', 'POST'])
def CD50():
    if request.method == 'POST':
        # Lấy giá trị lbkt từ form và kiểm tra tính hợp lệ
        try:
            lbkt = int(request.form['lbkt'])
        except ValueError:
            # Nếu lbkt không phải là số, yêu cầu nhập lại
            message = "Vui lòng nhập một số hợp lệ (1, 2 hoặc 3) cho loại bài kiểm tra."
            return render_template('index.html', message=message)

        # Kiểm tra lbkt chỉ là 1, 2 hoặc 3
        if lbkt not in [1, 2, 3]:
            message = "Loại bài kiểm tra không hợp lệ. Vui lòng nhập 1, 2 hoặc 3."
            return render_template('index.html', message=message)

        # Nếu lbkt hợp lệ, tiếp tục xử lý ảnh
        image_file = request.files['image']
        print(lbkt)

        # Lưu ảnh vào thư mục tạm thời
        image_path = os.path.join('static/uploads', image_file.filename)
        image_file.save(image_path)
        image_path = f'static/uploads/{image_file.filename}'

        # Gọi hàm cham_bai_kiem_tra
        result = cham_bai_kiem_tra(lbkt, image_path)

        # Kiểm tra kết quả trả về
        if result is None:
            message = "Có lỗi trong quá trình chấm điểm"
            return render_template('index.html', message=message)
        else:
            # Chấm điểm thành công, hiển thị thông tin đầy đủ
            return render_template('result.html', result=result)

    # Hiển thị trang index ban đầu
    return render_template('index.html')


@auth.route('/Chamdiem50NoUpdate', methods=['GET', 'POST'])
def CD50NoUpdate():
    from ChamDiem50WebNoUpdate import cham_bai_kiem_tra
    if request.method == 'POST':
        # Lấy giá trị lbkt từ form và kiểm tra tính hợp lệ
        try:
            lbkt = int(request.form['lbkt'])
        except ValueError:
            # Nếu lbkt không phải là số, yêu cầu nhập lại
            message = "Vui lòng nhập một số hợp lệ (1, 2 hoặc 3) cho loại bài kiểm tra."
            return render_template('index.html', message=message)

        # Kiểm tra lbkt chỉ là 1, 2 hoặc 3
        if lbkt not in [1, 2, 3]:
            message = "Loại bài kiểm tra không hợp lệ. Vui lòng nhập 1, 2 hoặc 3."
            return render_template('index.html', message=message)

        # Nếu lbkt hợp lệ, tiếp tục xử lý ảnh
        image_file = request.files['image']
        print(lbkt)

        # Lưu ảnh vào thư mục tạm thời
        image_path = os.path.join('static/uploads', image_file.filename)
        image_file.save(image_path)
        image_path = f'static/uploads/{image_file.filename}'

        # Gọi hàm cham_bai_kiem_tra
        result = cham_bai_kiem_tra(lbkt, image_path)

        # Kiểm tra kết quả trả về
        if result is None:
            message = "Có lỗi trong quá trình chấm điểm"
            return render_template('index2.html', message=message)
        else:
            # Chấm điểm thành công, hiển thị thông tin đầy đủ
            return render_template('result2.html', result=result)

    # Hiển thị trang index ban đầu
    return render_template('index2.html')


@auth.route('/Chamdiem120', methods=['GET', 'POST'])
def CD120():
    if request.method == 'POST':
        lbkt = int(request.form['lbkt'])

        image_file = request.files['image']
        print(lbkt)
        # Lưu ảnh vào thư mục tạm thời
        image_path = os.path.join('static/uploads', image_file.filename)

        image_file.save(image_path)
        image_path=f'static/uploads/{image_file.filename}'
        # Gọi hàm cham_bai_kiem_tra
        result = cham_bai_kiem_tra_120(lbkt, image_path)

        # Kiểm tra kết quả trả về
        if result is None:
            message = "Có lỗi trong quá trình chấm điểm"
            return render_template('index1.html', message=message)

        else:

            # Chấm điểm thành công, hiển thị thông tin đầy đủ
            return render_template('result1.html', result=result)

    return render_template('index1.html')



@auth.route('/Chamdiem120NoUpdate', methods=['GET', 'POST'])
def CD120NoUpdate():
    if request.method == 'POST':
        lbkt = int(request.form['lbkt'])
        from ChamDiem120WebNoUpdate import cham_bai_kiem_tra_120
        image_file = request.files['image']
        print(lbkt)
        # Lưu ảnh vào thư mục tạm thời
        image_path = os.path.join('static/uploads', image_file.filename)

        image_file.save(image_path)
        image_path=f'static/uploads/{image_file.filename}'
        # Gọi hàm cham_bai_kiem_tra
        result = cham_bai_kiem_tra_120(lbkt, image_path)

        # Kiểm tra kết quả trả về
        if result is None:
            message = "Có lỗi trong quá trình chấm điểm"
            return render_template('index3.html', message=message)

        else:

            # Chấm điểm thành công, hiển thị thông tin đầy đủ
            return render_template('result3.html', result=result)

    return render_template('index3.html')

# Route thêm bài kiểm tra (sẽ định nghĩa sau)
from flask import request, render_template, flash
from ThemBaiKiemTraWeb import nhapBaiKiemTra


@auth.route('/ThemBaiKiemTra', methods=['GET', 'POST'])
def them_bai_kiem_tra():
    ma_de = loai_baiKt = so_cau_hoi = dictDA = None

    if request.method == 'POST':
        # Lấy dữ liệu từ form
        ma_de = request.form.get('ma_de')
        loai_baiKt = int(request.form.get('loai_baiKt'))
        so_cau_hoi = int(request.form.get('so_cau_hoi'))
        dictDA = {}

        # Kiểm tra có đáp án từ điển không
        if 'dictDA' in request.form and request.form['dictDA']:
            try:
                dictDA = eval(request.form.get('dictDA'))

                # Kiểm tra số lượng key trong dictDA và định dạng
                if len(dictDA) != so_cau_hoi or not all(
                        isinstance(k, int) and isinstance(v, int) and v in [0, 1, 2, 3] for k, v in dictDA.items()):
                    flash("Lỗi: Số câu hỏi không trùng hoặc định dạng đáp án không đúng.", "error")
                    return render_template('them_bai_kiem_tra.html',
                                           ma_de=ma_de, loai_baiKt=loai_baiKt, so_cau_hoi=so_cau_hoi,
                                           error="Lỗi: Số câu hỏi không trùng hoặc định dạng đáp án không đúng.")
            except Exception:
                flash("Lỗi: Định dạng đáp án không hợp lệ.", "error")
                return render_template('them_bai_kiem_tra.html',
                                       ma_de=ma_de, loai_baiKt=loai_baiKt, so_cau_hoi=so_cau_hoi,
                                       error="Lỗi: Định dạng đáp án không hợp lệ.")
        else:
            # Nếu không có từ điển đáp án, lấy lựa chọn từ form
            for i in range(1, so_cau_hoi + 1):
                answer = request.form.get(f'answer_{i}')
                if answer is None or answer not in ['0', '1', '2', '3']:
                    flash("Lỗi: Bạn phải chọn đáp án cho từng câu hỏi.", "error")
                    return render_template('them_bai_kiem_tra.html',
                                           ma_de=ma_de, loai_baiKt=loai_baiKt, so_cau_hoi=so_cau_hoi)
                dictDA[i] = int(answer)

        # Gọi hàm nhapBaiKiemTra và xử lý kết quả
        status, console_output = nhapBaiKiemTra(ma_de, loai_baiKt, so_cau_hoi, dictDA)
        if status == 'Failed to add BKT':
            return render_template('them_bai_kiem_tra.html',
                                   ma_de=ma_de, loai_baiKt=loai_baiKt, so_cau_hoi=so_cau_hoi,
                                   error=console_output.replace("<br>", "\n"))

        # Thành công thì chuyển sang trang thông báo thành công
        return render_template('success.html', message="Thêm bài kiểm tra thành công")

    # Render trang với các giá trị ban đầu nếu không phải POST
    return render_template('them_bai_kiem_tra.html')


# Route thống kê điểm (sẽ định nghĩa sau)
import pyodbc
from flask import render_template


@auth.route('/ThongKeDiem', methods=['GET', 'POST'])
def ThongKeDiem():
    # Thiết lập kết nối với cơ sở dữ liệu SQL Server
    conn = pyodbc.connect(
        'Server=DESKTOP-AMN3OI8;Database=SinhVienVaDiem;UID=sa;PWD=051120;PORT=1433;DRIVER={SQL Server}'
    )

    # Tạo một con trỏ để thực thi câu lệnh SQL
    cursor = conn.cursor()

    # Thực hiện câu lệnh SELECT để lấy tất cả bản ghi từ bảng QuanLyDiem
    cursor.execute("SELECT * FROM QuanLyDiem")

    # Lấy tất cả các bản ghi trả về
    records = cursor.fetchall()

    # Đóng kết nối
    conn.close()

    # Trả về template và truyền dữ liệu vào template
    return render_template('Thongkediem.html', records=records)


# Route trang chính (homepage)
@auth.route('/')
def homepage():
    return render_template('homepage.html')


import pyodbc
from flask import render_template


@auth.route('/viewAllBkt', methods=['GET'])
def viewAllBkt():
    # Kết nối đến cơ sở dữ liệu SQL Server
    conn = pyodbc.connect(
        'Server=DESKTOP-AMN3OI8;Database=SinhVienVaDiem;UID=sa;PWD=051120;PORT=1433;DRIVER={SQL Server}'
    )

    # Tạo một con trỏ để thực thi câu lệnh SQL
    cursor = conn.cursor()

    # Thực hiện câu lệnh SELECT để lấy tất cả các bài kiểm tra
    cursor.execute("SELECT * FROM BaiKiemTra")

    # Lấy tất cả các bản ghi trả về
    records = cursor.fetchall()

    # Đóng kết nối
    conn.close()

    # Hàm chuyển đổi giá trị sau dấu ":" thành chữ cái
    def convert_to_letter(value):
        mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        return mapping.get(value, str(value))  # Trả lại chữ cái nếu có trong mapping, ngược lại trả lại giá trị ban đầu

    # Hàm chuyển đổi cột record[3] thành dạng 1:A,2:C...
    def convert_record(record_str):
        parts = record_str.split(',')
        converted_parts = []
        for part in parts:
            question, answer = part.split(':')
            # Chuyển đổi số sau dấu ':' thành chữ cái, không cần ép kiểu int cho answer nữa
            answer_int = int(answer)  # Đây chỉ thực hiện nếu `answer` là một số hợp lệ
            converted_parts.append(f"{question}:{convert_to_letter(answer_int)}")
        return ','.join(converted_parts)

    # Nhóm bài kiểm tra theo loại
    grouped_records = {1: [], 2: [], 3: []}
    for record in records:
        # Chuyển đổi cột record[3] (giả sử record[3] chứa chuỗi như "1:0,2:2,3:0,4:2,5:2,6:1")
        record_str = record[3]
        converted_record_str = convert_record(record_str)
        # Cập nhật lại record với chuỗi đã chuyển đổi
        record = (record[0], record[1], record[2], converted_record_str)  # Cập nhật lại record

        # Nhóm bài kiểm tra theo loai_bai_kiem_tra (tức là record[1])
        loai_bai_kiem_tra = record[1]  # Lấy giá trị loai_bai_kiem_tra từ record[1]
        if loai_bai_kiem_tra not in grouped_records:
            grouped_records[loai_bai_kiem_tra] = []  # Khởi tạo danh sách rỗng cho khóa
        grouped_records[loai_bai_kiem_tra].append(record)

        grouped_records[loai_bai_kiem_tra].append(record)

    # Trả về template và truyền dữ liệu vào template
    return render_template('view_all_bkt.html', grouped_records=grouped_records)






# Route để upload ảnh và chấm điểm theo folder
@auth.route('/ChamdiemTheoFolder', methods=['GET', 'POST'])
def ChamdiemTheoFolder():
    from werkzeug.utils import secure_filename
    import os
    if request.method == 'POST':
        from ChamDiemTheoFolderWeb import ChamTheoFolder
        # Lấy loại bài kiểm tra từ form
        lbkt = int(request.form['lbkt'])

        # Xóa tất cả các ảnh cũ trong thư mục static/uploadFolder
        upload_folder = current_app.config['UPLOAD_FOLDER']
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Lưu các ảnh mới được upload vào thư mục static/uploadFolder
        files = request.files.getlist('images')
        for file in files:
            if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filename = secure_filename(file.filename)
                file.save(os.path.join(upload_folder, filename))

        # Gọi hàm ChamTheoFolder để chấm điểm ảnh trong folder đã upload
        result = ChamTheoFolder(lbkt, upload_folder)

        # Hiển thị kết quả trên template resultChamFolder.html
        return render_template('resultChamFolder.html', result=result)

    # Nếu là GET, hiển thị form để upload ảnh
    return render_template('upload_folder.html')

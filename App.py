# App.py
from flask import Flask

def create_app():
    app = Flask(__name__)
    app.secret_key = 'your_secret_key'
    app.config['DEBUG'] = True
    app.config['PROPAGATE_EXCEPTIONS'] = True
    # Định nghĩa đường dẫn để lưu ảnh tải lên
    UPLOAD_FOLDER = 'static/uploadFolder'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    # Không sử dụng CSRF protection
    # Không cần thêm CSRFProtect(app)

    # Import và đăng ký blueprint cho các route xác thực
    from auth_routes import auth
    app.register_blueprint(auth)  # Thêm tiền tố /auth cho các route

    return app


from flask import Flask
from routes import register_routes  # Import hàm đăng ký các routes

app = Flask(__name__)

# Đăng ký các routes từ file routes.py
register_routes(app)

# Trang chủ
@app.route('/')
def home():
    return "Welcome to the Insurance Prediction API"

if __name__ == '__main__':
    app.run(debug=True, port=4000)  # Khởi động ứng dụng trên cổng 4000

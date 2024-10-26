# Dự án Dự đoán Chi phí Y tế

Mô tả dự án
Dự án này sử dụng một mô hình học máy để dự đoán chi phí y tế dựa trên các thông tin cá nhân như tuổi, chỉ số BMI, số lượng con, tình trạng hút thuốc, và vùng địa lý. API được xây dựng trên Flask và có thể xử lý các thao tác như huấn luyện mô hình, dự đoán chi phí y tế, và đánh giá mô hình.

Cấu trúc dự án

.
├── app.py                  # File chính để khởi chạy ứng dụng Flask
├── routes.py               # Các API endpoint chính của ứng dụng
├── models/                 # Thư mục chứa các mô hình đã huấn luyện
│   ├── trained_model.pkl    # Mô hình đã được huấn luyện
│   ├── test_data.csv        # Dữ liệu kiểm tra đã được tách ra trong quá trình huấn luyện
│   └── trained_columns.pkl  # Danh sách các cột đã được mã hóa trong quá trình huấn luyện
├── data/                   # Thư mục chứa dữ liệu thô
│   └── insurance.csv        # Dữ liệu insurance.csv
├── eda.py                  # File phân tích dữ liệu (Exploratory Data Analysis)
├── engine.py               # File xử lý dữ liệu
├── model_performance.py     # File đánh giá và huấn luyện mô hình
├── stats.py                # File tính toán các thông số thống kê
├── utils.py                # Các hàm hỗ trợ
└── README.md               # File README (hướng dẫn sử dụng)

## Yêu cầu hệ thống

Python 3.x

## Các thư viện cần thiết (được cài đặt qua requirements.txt)

Cài đặt
Clone repository:

git clone https://github.com/your-repository/medical-cost-prediction.git
cd medical-cost-prediction

## Tạo môi trường ảo (virtual environment):

python -m venv venv
source venv/bin/activate  # Trên macOS/Linux
venv\Scripts\activate      # Trên Windows

## Cài đặt các thư viện cần thiết:

pip install -r requirements.txt

## Khởi động ứng dụng Flask:

python app.py
Ứng dụng sẽ chạy trên địa chỉ http://127.0.0.1:4000

1. POST /api/eda - Phân tích dữ liệu (Exploratory Data Analysis)
Test Case 1: Dữ liệu hợp lệ
Mục đích: Đảm bảo API hoạt động đúng với dữ liệu hợp lệ.
URL: http://127.0.0.1:4000/api/eda
Method: POST
Request Body (JSON):

[
  {"age": 25, "sex": "female", "bmi": 27.9, "children": 0, "smoker": "yes", "region": "southwest", "charges": 16884.924},
  {"age": 30, "sex": "male", "bmi": 22.9, "children": 1, "smoker": "no", "region": "southeast", "charges": 1725.5523}
]

1. POST /api/train - Huấn luyện mô hình
Test Case 3: Huấn luyện mô hình với dữ liệu hợp lệ
Mục đích: Đảm bảo mô hình được huấn luyện thành công với dữ liệu hợp lệ.
URL: http://127.0.0.1:4000/api/train
Method: POST
Request Body (JSON):

[
  {"age": 25, "sex": "female", "bmi": 27.9, "children": 0, "smoker": "yes", "region": "southwest", "charges": 16884.924},
  {"age": 30, "sex": "male", "bmi": 22.9, "children": 1, "smoker": "no", "region": "southeast", "charges": 1725.5523},
  {"age": 35, "sex": "female", "bmi": 30.0, "children": 2, "smoker": "no", "region": "northwest", "charges": 4449.462}
]

1. POST /api/predict - Dự đoán
Test Case 5: Dự đoán với dữ liệu hợp lệ
Mục đích: Đảm bảo API trả về kết quả dự đoán chính xác với dữ liệu hợp lệ.
URL: http://127.0.0.1:4000/api/predict
Method: POST
Request Body (JSON):

{
  "age": 35,
  "sex": "male",
  "bmi": 25.9,
  "children": 1,
  "smoker": "no",
  "region": "southeast"
}

1. PUT /api/update-model - Cập nhật mô hình
Test Case 7: Cập nhật mô hình với dữ liệu mới
Mục đích: Đảm bảo rằng API cập nhật mô hình với dữ liệu mới thành công.
URL: http://127.0.0.1:4000/api/update-model
Method: PUT
Request Body (JSON):

[
  {"age": 45, "sex": "female", "bmi": 29.9, "children": 3, "smoker": "no", "region": "northwest", "charges": 23456.78},
  {"age": 55, "sex": "male", "bmi": 32.0, "children": 2, "smoker": "yes", "region": "northeast", "charges": 43765.98}
]

5. GET /api/evaluate - Đánh giá mô hình
Test Case 9: Đánh giá mô hình với dữ liệu kiểm tra đã lưu
Mục đích: Đảm bảo API trả về kết quả đánh giá mô hình trên tập dữ liệu kiểm tra.
URL: http://127.0.0.1:4000/api/evaluate
Method: GET
Expected Response:

{
  "mean_squared_error": 3450.67
}


1. GET /api/stats - Thống kê mô hình
Test Case 11: Trả về thống kê mô hình hiện tại
Mục đích: Đảm bảo API trả về các thông tin thống kê cơ bản về mô hình đã được huấn luyện.
URL: http://127.0.0.1:4000/api/stats
Method: GET
Expected Response:

{
  "model_type": "Linear Regression",
  "total_params": 10,
  "training_samples": 1338
}
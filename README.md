# Decision Tree Student Outcome Prediction

## Mô tả

Dự án sử dụng mô hình cây quyết định để dự đoán kết quả học tập của sinh viên (Tốt nghiệp, Bỏ học, Đang theo học) dựa trên các thông tin cá nhân, học tập, tài chính, gia đình và kinh tế xã hội.

## Cấu trúc thư mục

```
decision-tree-project/
│
├── data/
│   └── data.csv                # Dữ liệu đầu vào (các thuộc tính sinh viên + cột target)
│
├── output/
│   └── predictions.csv         # Kết quả dự đoán của mô hình
│
├── src/
│   ├── DecisionTree.py         # Cài đặt thuật toán cây quyết định
│   ├── train.py                # Script huấn luyện và lưu mô hình
│   └── evaluate.py             # Script đánh giá và xuất kết quả dự đoán
│
└── Thông tin cá nhân.txt       # Mô tả chi tiết các thuộc tính dữ liệu
```

## Hướng dẫn sử dụng

### 1. Cài đặt đầy đủ các thư viện cần thiết

```sh
pip install pandas numpy scikit-learn
```

Nếu bạn sử dụng các thư viện khác (ví dụ: matplotlib để vẽ biểu đồ), hãy cài thêm:

```sh
pip install matplotlib
```

### 2. Huấn luyện mô hình

```sh
cd src
python train.py
```

Sau khi chạy xong, mô hình sẽ được lưu vào `decision_tree_model.pkl`.

### 3. Đánh giá và xuất kết quả dự đoán

```sh
python evaluate.py
```

- Kết quả dự đoán sẽ được lưu vào file `../output/predictions.csv`.
- Script sẽ in ra độ chính xác và phân bố dự đoán trên tập test.

### 4. Ý nghĩa output

- **predictions.csv** chứa: các thuộc tính đầu vào, nhãn thực tế (`TrueLabel`), nhãn dự đoán (`PredictedLabel`).
- Các giá trị nhãn: `Graduate`, `Dropout`, `Enrolled`.

## Ghi chú

- Đảm bảo file `data.csv` có đầy đủ các cột như mô tả trong `Thông tin cá nhân.txt`.
- Nếu thay đổi cấu trúc dữ liệu, hãy cập nhật lại danh sách thuộc tính trong code.

---

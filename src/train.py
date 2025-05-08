from DecisionTree import id3
import csv
import os
import pprint
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def load_data(filepath):
    df = pd.read_csv(filepath)
    # Kiểm tra dữ liệu thiếu
    print("Số lượng giá trị thiếu mỗi cột:")
    print(df.isnull().sum())
    # Nếu có thể, điền giá trị thiếu (ở đây điền bằng mode)
    df = df.fillna(df.mode().iloc[0])
    features = list(df.columns[:-1])
    samples = df.values.tolist()
    return samples, features, df

def visualize_data(df):
    # Vẽ phân phối nhãn
    plt.figure(figsize=(6,4))
    df[df.columns[-1]].value_counts().plot(kind='bar')
    plt.title('Phân phối nhãn')
    plt.xlabel('Nhãn')
    plt.ylabel('Số lượng')
    plt.tight_layout()
    plt.show()

    # Vẽ phân phối cho từng thuộc tính rời rạc (categorical)
    # for col in df.columns[:-1]:
    #     plt.figure(figsize=(6,4))
    #     df[col].value_counts().plot(kind='bar')
    #     plt.title(f'Phân phối thuộc tính: {col}')
    #     plt.xlabel(col)
    #     plt.ylabel('Số lượng')
    #     plt.tight_layout()
    #     plt.show()

if __name__ == "__main__":
    data_path = os.path.join('..', 'data', 'data.csv')
    if not os.path.exists(data_path):
        print(f"File {data_path} không tồn tại.")
        exit(1)

    data, features, df = load_data(data_path)
    print("Một vài dòng dữ liệu đầu tiên:")
    print(df.head())

    visualize_data(df)

    tree = id3(data, features)
    print("Cây quyết định huấn luyện được:")
    pprint.pprint(tree, width=120, sort_dicts=False)

    # Lưu mô hình
    with open('decision_tree_model.pkl', 'wb') as f:
        pickle.dump(tree, f)
    print("Đã lưu mô hình vào decision_tree_model.pkl")


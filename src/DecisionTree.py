import math
from collections import Counter

def split_data(data, feature_index, value):
    "Chia dữ liệu thành một tập con dựa trên giá trị của một thuộc tính"
    return [row for row in data if row[feature_index] == value]

def entropy(labels):
    "Tính entropy của một danh sách nhãn"
    label_counts = Counter(labels)
    total_count = len(labels)
    return -sum((count / total_count) * math.log2(count / total_count) for count in label_counts.values())

def information_gain(data, feature_index):
    "Tính Information từ việc chia dữ liệu theo một thuộc tính"
    total_labels = [row[-1] for row in data]
    total_entropy = entropy(total_labels)

    values = set(row[feature_index] for row in data)
    weighted_entropy = 0.0

    for value in values:
        subset = split_data(data, feature_index, value)
        subset_labels = [row[-1] for row in subset]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset_labels)

    return total_entropy - weighted_entropy

def majority_class(data):
    "Tìm lớp chiếm ưu thế trong một danh sách nhãn"
    label_counts = Counter(row[-1] for row in data)
    return label_counts.most_common(1)[0][0]

def id3(data, features):
    "Xây dựng cây quyết định bằng thuật toán ID3"
    labels = [row[-1] for row in data]

    # Trường hợp tất cả nhãn giống nhau
    if labels.count(labels[0]) == len(labels):
        return labels[0]

    # Không còn thuộc tính để chia
    if not features:
        return majority_class(data)

    # Tính gain cho từng thuộc tính
    gains = [information_gain(data, i) for i in range(len(features))]
    best_index = gains.index(max(gains))
    best_feature = features[best_index]

    tree = {best_feature: {}}
    feature_values = set(row[best_index] for row in data)

    for value in feature_values:
        subset = split_data(data, best_index, value)
        if not subset:
            tree[best_feature][value] = majority_class(data)
        else:
            # Loại bỏ cột tương ứng
            new_features = features[:best_index] + features[best_index+1:]
            new_data = [row[:best_index] + row[best_index+1:] for row in subset]
            tree[best_feature][value] = id3(new_data, new_features)

    return tree

def predict(tree, sample, feature_to_index, default=None):
    "Dự đoán lớp của một mẫu dựa trên cây quyết định"
    while isinstance(tree, dict):
        feature = next(iter(tree))
        feature_index = feature_to_index[feature]
        feature_value = sample[feature_index]
        if feature_value in tree[feature]:
            tree = tree[feature][feature_value]
        else:
            return default
    return tree

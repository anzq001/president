import numpy as np
import os
label_dict = {"克林顿": 0, "奥巴马": 1, "特朗普": 2}


def get_filenames(file_dir, ratio):
    file_list = []
    label_list = []
    for label_dir in os.listdir(file_dir):
        for filename in os.listdir(os.path.join(file_dir, label_dir)):
            file_list.append(os.path.join(os.path.join(file_dir, label_dir, filename)))
            label_list.append(label_dict[label_dir])

    # 利用shuffle打乱顺序
    all_data = np.array([file_list, label_list]).transpose()
    np.random.shuffle(all_data)
    train_num = int(np.ceil(ratio * len(all_data)))
    train = all_data[0:train_num, :]
    test = all_data[train_num:-1, :]
    train_filename = list(train[:, 0])
    train_label = [int(i) for i in list(train[:, 1])]
    test_filename = list(test[:, 0])
    test_label = [int(i) for i in list(test[:, 1])]
    return train_filename, train_label, test_filename, test_label

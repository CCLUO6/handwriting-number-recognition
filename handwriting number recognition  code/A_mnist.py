# coding: utf-8
train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

import os.path
import gzip
import pickle
import os
import numpy as np
import logging
import wget
from pathlib import Path

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__)) + '/dataset'
save_file = dataset_dir + "/mnist.pkl"

logging.basicConfig(level=logging.INFO, format="%(message)s")

def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data

def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset

def download_minst(save_dir: str = None) -> bool:
    """下载MNIST数据集
    输入参数:
        save_dir: MNIST数据集的保存地址. 类型: 字符串
    返回值:
        全部下载成功返回True, 否则返回False
    """

    save_dir = Path(save_dir)
    train_set_imgs_addr = save_dir / "train-images-idx3-ubyte.gz"
    train_set_labels_addr = save_dir / "train-labels-idx1-ubyte.gz"
    test_set_imgs_addr = save_dir / "t10k-images-idx3-ubyte.gz"
    test_set_labels_addr = save_dir / "t10k-labels-idx1-ubyte.gz"

    try:
        if not os.path.exists(train_set_imgs_addr):
            logging.info("下载train-images-idx3-ubyte.gz")
            filename = wget.download(url="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                                     out=str(train_set_imgs_addr))
            logging.info("\tdone.")
        else:
            logging.info("train-images-idx3-ubyte.gz已经存在.")

        if not os.path.exists(train_set_labels_addr):
            logging.info("下载train-labels-idx1-ubyte.gz.")
            filename = wget.download(url="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                                     out=str(train_set_labels_addr))
            logging.info("\tdone.")
        else:
            logging.info("train-labels-idx1-ubyte.gz已经存在.")

        if not os.path.exists(test_set_imgs_addr):
            logging.info("下载t10k-images-idx3-ubyte.gz.")
            filename = wget.download(url="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                                     out=str(test_set_imgs_addr))
            logging.info("\tdone.")
        else:
            logging.info("t10k-images-idx3-ubyte.gz已经存在.")

        if not os.path.exists(test_set_labels_addr):
            logging.info("下载t10k-labels-idx1-ubyte.gz.")
            filename = wget.download(url="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
                                     out=str(test_set_labels_addr))
            logging.info("\tdone.")
        else:
            logging.info("t10k-labels-idx1-ubyte.gz已经存在.")

    except:
        return False

    return True

def init_mnist():
    download_minst(save_dir="dataset")
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")
    train_img_count = len(dataset['train_img'])
    print("train_img数量:", train_img_count)
    test_img_count = len(dataset['test_img'])
    print("test_img数量:", test_img_count)


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(normalize=True, one_hot_label=False, flatten=False):
    """读入MNIST数据集

    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label :
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组

    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    init_mnist()


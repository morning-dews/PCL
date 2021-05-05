import os
import tarfile
import numpy as np
import torch
import helper.cl_dataset as cl_dataset
import helper.file_helper as file_manager
import pdb
import matplotlib.pyplot as plt
from scipy import misc

__author__ = 'garrett_local'


def _binarize(labels):
    return ((labels == 0) | (labels == 1) | (labels == 8) |  # (labels == 4) |
            (labels == 9)) * 1  # * 2 - 1 (labels == 4) |


def _prepare_cifar100_data():
    data_path = '/home/huwenp/Dataset/CIFAR/'
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    file_manager.create_dirname_if_not_exist(data_path)
    file_name = os.path.basename(url)
    full_path = os.path.join(data_path, file_name)
    folder = os.path.join(data_path, 'cifar-10-batches-py-feature')
    if not os.path.isdir(folder):
        file_manager.download(url, data_path)
        with tarfile.open(full_path) as f:
            f.extractall(path=data_path)
    train_x = []
    train_y = []
    for i in range(0, 5):
        file_path = os.path.join(folder, 'traindata_batch_{0:d}.pt'.format(i))
        data_dict = file_manager.unpickle(file_path)
        pdb.set_trace()
        train_x.append(data_dict['data'])
        train_y.append(data_dict['labels'])
    train_x = np.concatenate(train_x) / 255.0 
    train_y = np.concatenate(train_y)

    data_dict = file_manager.unpickle(os.path.join(folder, 'testdata_batch_0.pt'))
    test_x = data_dict['data'] / 255.0
    test_y = np.array(data_dict['labels'])
    # pdb.set_trace()

    # .transpose([0, 2, 3, 1])
    train_x = train_x.reshape((train_x.shape[0], 3, 32, 32))
    # .transpose([0, 2, 3, 1])
    test_x = test_x.reshape((test_x.shape[0], 3, 32, 32))
    
    # for i in range(10):
    #     # pdb.set_trace()
    #     misc.imsave('./images/cifar10' + str(train_y[i]) + '-' + str(i) + '.jpg', train_x[i].transpose(1, 2, 0))
    
    # pdb.set_trace()
    # train_y = _binarize(train_y)
    # test_y = _binarize(test_y)
    return train_x, train_y, test_x, test_y

def _prepare_cifar100f_data():
    data_path = '/home/huwenp/Dataset/CIFAR100/'
    # url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    # file_manager.create_dirname_if_not_exist(data_path)
    # file_name = os.path.basename(url)
    # full_path = os.path.join(data_path, file_name)
    folder = os.path.join(data_path, 'features')
    train_x = []
    train_y = []
    for i in range(0, 5):
        file_path = os.path.join(folder, 'traindata_batch_{0:d}.pt'.format(i))
        data_dict = torch.load(file_path)
        # pdb.set_trace()
        train_x.append(data_dict['data'])
        train_y.append(data_dict['labels'])
    train_x = np.concatenate(train_x)

    # pdb.set_trace()
    # train_x = (train_x - train_x.mean(1).reshape(-1, 1)) / np.linalg.norm(train_x, axis=1, keepdims=True)
    train_x = train_x / np.linalg.norm(train_x, axis=1, keepdims=True) - 0.02

    train_y = np.concatenate(train_y)
    # pdb.set_trace()

    data_dict = torch.load(os.path.join(folder, 'testdata_batch_0.pt'))
    test_x = data_dict['data']
    # test_x = (test_x - test_x.mean(1).reshape(-1, 1)) / np.linalg.norm(test_x, axis=1, keepdims=True)
    test_x = test_x / np.linalg.norm(test_x, axis=1, keepdims=True) - 0.02

    test_y = np.array(data_dict['labels'])

    # .transpose([0, 2, 3, 1])
    # train_x = train_x.reshape((train_x.shape[0], 3, 32, 32))
    # # .transpose([0, 2, 3, 1])
    # test_x = test_x.reshape((test_x.shape[0], 3, 32, 32))
    
    # for i in range(10):
    #     # pdb.set_trace()
    #     misc.imsave('./images/cifar10' + str(train_y[i]) + '-' + str(i) + '.jpg', train_x[i].transpose(1, 2, 0))
    
    # pdb.set_trace()
    # train_y = _binarize(train_y)
    # test_y = _binarize(test_y)
    return train_x, train_y, test_x, test_y


class Cifar100Dataset(cl_dataset.CLClassDataSet):

    def __init__(self, *args, **kwargs):
        self._train_x, self._train_y, self._test_x, self._test_y = \
            _prepare_cifar100f_data()
        super(Cifar100Dataset, self).__init__(*args, **kwargs)
        self.data_size = 0
        self.test_size = self._test_y.shape[0]
    def _original_train_x(self):
        return self._train_x

    def _original_train_y(self):
        return self._train_y

    def _original_test_x(self):
        return self._test_x

    def _original_test_y(self):
        return self._test_y

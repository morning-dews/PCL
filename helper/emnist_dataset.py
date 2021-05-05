from sklearn.datasets import fetch_mldata
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
import torchvision
import pdb
import helper.cl_dataset as cl_dataset
from scipy import io

__author__ = 'garrett_local'


def _prepare_emnist_data():
    # emnist = fetch_mldata('MNIST original', data_home='/home/huwenp/Dataset/EMNIST/')
    # # mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    # # mnist = torchvision.datasets.MNIST('../../../Data/',train=True,download=True)
    # x = emnist.data
    # y = emnist.target
    # pdb.set_trace()
    # # np.mean(x, 1).reshape(-1, 1) / 255.0
    # # / np.mean(x, 1).reshape(-1, 1)
    # x = np.reshape(x, (x.shape[0], -1)) / 255.0
    # train_x = np.asarray(x[:60000], dtype=np.float32)
    # train_y = np.asarray(y[:60000], dtype=np.int32)
    # test_x = np.asarray(x[60000:], dtype=np.float32)
    # test_y = np.asarray(y[60000:], dtype=np.int32)
    mnist = io.loadmat('/home/huwenp/Dataset/EMNIST/mldata/mnist-original.mat', struct_as_record = True)
    train_x = np.asarray(mnist['dataset']['train'][0][0][0][0][0], dtype = np.float32) / 255.0 / 1.0 
    train_x = train_x / np.linalg.norm(train_x, axis=1, keepdims=True) - 0.01
    # pdb.set_trace()

    train_y = np.asarray(mnist['dataset']['train'][0][0][0][0][1], dtype = np.int32).squeeze()
    test_x = np.asarray(mnist['dataset']['test'][0][0][0][0][0], dtype = np.float32) / 255.0 /1.0 
    test_x = test_x / np.linalg.norm(test_x, axis=1, keepdims=True) - 0.01
    test_y = np.asarray(mnist['dataset']['test'][0][0][0][0][1], dtype = np.int32).squeeze()
    # pdb.set_trace()


    return train_x, train_y, test_x, test_y


class EMnistDataset(cl_dataset.CLClassDataSet):

    def __init__(self, *args, **kwargs):
        self._train_x, self._train_y, self._test_x, self._test_y = \
            _prepare_emnist_data()
        super(EMnistDataset, self).__init__(*args, **kwargs)

        self.test_size = self._test_y.shape[0]
    def _original_train_x(self):
        return self._train_x

    def _original_train_y(self):
        return self._train_y

    def _original_test_x(self):
        return self._test_x

    def _original_test_y(self):
        return self._test_y

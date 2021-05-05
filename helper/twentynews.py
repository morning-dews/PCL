# from sklearn.datasets import fetch_mldata
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
import torchvision
from sklearn.feature_extraction.text import TfidfVectorizer
import pdb
import helper.cl_dataset as cl_dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline

__author__ = 'garrett_local'


def _prepare_twentynews_data():
    tnews_train = fetch_20newsgroups(data_home='../../../Data/twentynews/',subset='train', remove=('headers','footers','quotes'))
    tnews_test = fetch_20newsgroups(data_home='../../../Data/twentynews/',subset='test', remove=('headers','footers','quotes'))
    # tnews_train = fetch_20newsgroups(data_home='../../../Data/twentynews/',subset='train')
    # tnews_test = fetch_20newsgroups(data_home='../../../Data/twentynews/',subset='test')
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    # mnist = torchvision.datasets.MNIST('../../../Data/',train=True,download=True)
    train_texts=tnews_train['data']
    train_y=tnews_train['target']
    test_texts=tnews_test['data']
    test_y=tnews_test['target']

    vectorizer = TfidfVectorizer(max_features=2000)
    train_x = vectorizer.fit_transform(train_texts).todense()
    test_x = vectorizer.transform(test_texts).todense()
    # np.mean(x, 1).reshape(-1, 1) / 255.0
    # / np.mean(x, 1).reshape(-1, 1)

    return train_x, train_y, test_x, test_y


class twentynews(cl_dataset.CLClassDataSet):

    def __init__(self, *args, **kwargs):
        self._train_x, self._train_y, self._test_x, self._test_y = \
            _prepare_twentynews_data()
        super(twentynews, self).__init__(*args, **kwargs)

        self.test_size = self._test_y.shape[0]
    def _original_train_x(self):
        return self._train_x

    def _original_train_y(self):
        return self._train_y

    def _original_test_x(self):
        return self._test_x

    def _original_test_y(self):
        return self._test_y

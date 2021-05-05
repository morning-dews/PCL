##Version 1.0 Note all
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


class DBPedia(cl_dataset.CLDBPediaDataSet):

    def __init__(self, *args, **kwargs):
        super(DBPedia, self).__init__(*args, **kwargs)
        test_x, test_y = self._prepare_dbpedia_test_data()

        self.test_size = test_y.shape[0]

    def _original_train_x(self):
        return self._train_x

    def _original_train_y(self):
        return self._train_y

    def _original_test_x(self):
        return self._test_x

    def _original_test_y(self):
        return self._test_y

    def _prepare_dbpedia_train_data(self, index):
        data_path = '/home/huwenp/project/ContinualLearning/Data/dbpedia/TFIDF'

        file_path = os.path.join(data_path, 'train_data{0:d}.pt'.format(index))
        train_x = torch.load(file_path)
        # pdb.set_trace()
        # train_x = train_x / np.linalg.norm(train_x, axis=1, keepdims=True)


        return train_x

    def _prepare_dbpedia_test_data(self):
        data_path = '/home/huwenp/project/ContinualLearning/Data/dbpedia/TFIDF'
        test_x = torch.load(os.path.join(data_path, 'test_data.pt'))
        # test_x = data_dict['data']
        # test_x = test_x / np.linalg.norm(test_x, axis=1, keepdims=True)
        test_y = torch.load(os.path.join(data_path, 'test_label.pt')) - 1

        return test_x, test_y









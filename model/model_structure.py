import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import pdb
import math
import copy


class Model_Structure(nn.Module):
    def __init__(self, models=None):
        super(Model_Structure, self).__init__()
        self.weight1 = None if models is None else copy.deepcopy(models.weight1)
        self.weight2 = None if models is None else copy.deepcopy(models.weight2)
        self.weight3 = None if models is None else copy.deepcopy(models.weight3)
        self.ready = False
        self.none = True

    def share_mean(self, weights):
        self.ready = True
        self.__init__(weights[0])
        for temp_index in range(1, len(weights)):
            self.weight1.data = self.weight1.data + weights[temp_index].weight1.data
            self.weight2.data = self.weight2.data + weights[temp_index].weight2.data
            self.weight3.data = self.weight3.data + weights[temp_index].weight3.data

        self.weight1.data = self.weight1.data / len(weights)
        self.weight2.data = self.weight2.data / len(weights)
        self.weight3.data = self.weight3.data / len(weights)

    def addtoself(self, weights):

        self.ready = True
        if self.none:
            self.none = False
            self.weight1 = copy.deepcopy(weights.weight1)
            self.weight2 = copy.deepcopy(weights.weight2)
            self.weight3 = copy.deepcopy(weights.weight3)
        else:
            self.weight1.data = self.weight1.data + weights.weight1.data
            self.weight2.data = self.weight2.data + weights.weight2.data
            self.weight3.data = self.weight3.data + weights.weight3.data

    def mmtoself(self, weights):

        self.ready = True
        if self.none:
            self.none = False
            self.weight1 = copy.deepcopy(weights.weight1)
            self.weight2 = copy.deepcopy(weights.weight2)
            self.weight3 = copy.deepcopy(weights.weight3)
        else:
            self.weight1.data = self.weight1.data * weights.weight1.data
            self.weight2.data = self.weight2.data * weights.weight2.data
            self.weight3.data = self.weight3.data * weights.weight3.data

    def addmmtoself(self, weights1, weights2):

        self.ready = True
        if self.none:
            self.none = False
            self.weight1 = copy.deepcopy(weights2.weight1)
            self.weight2 = copy.deepcopy(weights2.weight2)
            self.weight3 = copy.deepcopy(weights2.weight3)

            self.weight1.data = weights1[0] * weights2.weight1.data
            self.weight2.data = weights1[1] * weights2.weight2.data
            self.weight3.data = weights1[2] * weights2.weight3.data
        else:
            self.weight1.data += weights1[0] * weights2.weight1.data
            self.weight2.data += weights1[1] * weights2.weight2.data
            self.weight3.data += weights1[2] * weights2.weight3.data




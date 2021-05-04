#Version 2.1 Note all
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from model.model_structure import Model_Structure
import pdb
import math
# from model.model_structure import Model_Structure
import copy
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
TensorB = torch.cuda.ByteTensor if cuda else torch.ByteTensor
TensorL = torch.cuda.LongTensor if cuda else torch.LongTensor

class SuCNN(nn.Module):
    def __init__(self, opt, label_index=None):
        super(SuCNN, self).__init__()

        self.opt = opt
        self.nepoch = opt.max_epochs
        self.cur_epoch = 0
        self.i_batch = 0
        self.out_num = opt.class_groups[label_index]

        dropoutrate = 0.1
        num_hidden_su = 1000
        self.su_model = nn.Sequential(
            nn.Linear(opt.img_size, num_hidden_su),
            nn.ReLU(),
            # nn.BatchNorm1d(self.shape[0]),
            # nn.Dropout(dropoutrate),
            # nn.Linear(num_hidden_su, num_hidden_su),
            # nn.ReLU(),
            # nn.BatchNorm1d(self.shape[0]),
            nn.Dropout(dropoutrate),
            ###################################
            nn.Linear(num_hidden_su, self.out_num)
        )


    def forward(self, img_ori):

        out = self.su_model(img_ori)

        return out








        
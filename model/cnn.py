##Version 2.1 Note all
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

class Recognizer_CNN(nn.Module):
    def __init__(self, opt, label_index=None):
        super(Recognizer_CNN, self).__init__()
        # dense1_bn = nn.BatchNorm1d(512)
        # dense2_bn = nn.BatchNorm1d(256)
        self.opt = opt
        self.nepoch = opt.max_epochs
        self.data_size = opt.data_size[label_index]
        self.cur_epoch = 0
        self.i_batch = 0
        # pdb.set_trace()
        self.false = False
        self.fisherMatrix = None
        self.LW = [[],[],[]]
        # if label_index is not None:
        self.class_groups = opt.class_groups
        self.out_num = 1
        self.share_model = Model_Structure()
        self.share_model_mode = Model_Structure()
        # else:
        #     self.out_num = 1
        self.dropout = nn.Dropout(opt.dropoutrate)
        self.dropout1 = nn.Dropout(opt.dropoutrate)
        self.eval_flag = False
        self.weight1 = nn.Parameter(torch.Tensor(100, opt.img_size))
        self.weight2 = nn.Parameter(torch.Tensor(100, 100))
        # self.bias1 = nn.Parameter(torch.Tensor(100))
        self.weight3 = nn.Parameter(torch.Tensor(3 + self.out_num, 100))

        self.SP = torch.autograd.Variable(torch.eye(opt.img_size).type(Tensor), requires_grad=False)
        self.P = torch.autograd.Variable(torch.eye(opt.img_size).type(Tensor), requires_grad=False)

        self.Pf = None

        # self.bias2 = nn.Parameter(torch.Tensor(2))
        # self.bias3 = nn.Parameter(torch.Tensor(100))
        # self.bias4 = nn.Parameter(torch.Tensor(784))

        self.reset_parameters()

    def projection(self, x, evals=False):
        eta = -1.0
        eta1 = 1.0
        squres = 29
        weights = 10.0
        if evals == True:
                x_new = x
                for i_p in range(1):
                    x_t = torch.mm(x_new, torch.t(self.SP.data))
                    lambda_i = x_new.norm() / (x_new.norm() + x_t.norm())

                    x_new = x_new  - x_t
        else:
            if self.Pf is None:
                x_new = x
            else:
                x_t = torch.mm(x, torch.t(self.Pf.data))
                x_new = x - x_t      

            
        return x_new
            # return eta * x_new + (1 - eta) * x
    def set_train_info(self, ibatch, current_epoch):
        self.i_batch = ibatch
        self.cur_epoch = current_epoch

    def get_parameters(self):
        return Model_Structure(self)

    def learn_p(self, x, alpha=0.000001): # alpha=0.000001
        # pro_weight(p, x, w, alpha=1.0, cnn=True, stride=1):
        lamda = self.i_batch / self.data_size / self.nepoch + self.cur_epoch/self.nepoch
        x = torch.mean(x, 0, True)
        # cnn = False
        # if cnn:
        #     _, _, H, W = x.shape
        #     F, _, HH, WW = w.shape
        #     S = stride  # stride
        #     Ho = int(1 + (H - HH) / S)
        #     Wo = int(1 + (W - WW) / S)
        #     for i in range(Ho):
        #         for j in range(Wo):
        #             # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
        #             r = x[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1)
        #             # r = r[:, range(r.shanipe[1] - 1, -1, -1)]
        #             k = torch.mm(self.P, torch.t(r))
        #             # pdb.set_trace()
        #             self.P.data.sub_(torch.mm(k, torch.t(k)) / (alpha ** lamda + torch.mm(r, k)))
        #     # w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(p.data)).view_as(w)
        # else:
        r = x
        k = torch.mm(self.P.data, torch.t(r))
        k1 = torch.mm(self.SP.data, torch.t(r))
        # pdb.set_trace()
        self.P.data.sub_(torch.mm(k, torch.t(k)) / (alpha ** lamda + 1.0 * torch.mm(r, k).data))
        self.SP.data.sub_(torch.mm(k1, torch.t(k1)) / (alpha ** lamda + 1.0 * torch.mm(r, k1).data))

    def set_projection(self, p):
        self.Pf = copy.deepcopy(p)
        # pass
        self.P = copy.deepcopy(p)

    def reset_parameters(self): 
        stdv = 1.0 / math.sqrt(1000)
        self.weight1.data.uniform_(-stdv, stdv)
        stdv = 1.0 / math.sqrt(500)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        # self.bias1.data.fill_(0)
        # self.bias2.data.fill_(0)
        # self.bias3.data.fill_(0)
        # self.bias4.data.fill_(0)

    def set_parameters(self, model):
        self.weight1.data = 1.0 * copy.deepcopy(model.weight1.data)
        self.weight2.data = 1.0 * copy.deepcopy(model.weight2.data)
        self.weight3.data = 1.0 * copy.deepcopy(model.weight3.data)

    def forward_decode(self, img):
        # pdb.set_trace()

        layer1 = torch.sigmoid(F.linear(img, self.weight3.transpose(0,1)))
        # layer1 = torch.relu(F.linear(img, self.weight2.transpose(0,1), self.bias3))
        # layer1 = self.dropout1(layer1)

        layer2 = torch.relu(F.linear(layer1, self.weight2.transpose(0,1)))
        # # layer1 = torch.relu(F.linear(img, self.weight2.transpose(0,1), self.bias3))
        # layer2 = self.dropout1(layer2)

        out = torch.sigmoid(F.linear(layer1, self.weight1.transpose(0,1)))
        # out = self.dropout1(out)
        # out = torch.sigmoid(F.linear(layer1, self.weight1.transpose(0,1), self.bias4))

        return out

    def forward(self, img_ori):
        # self.learn_p(img_ori)
        # if self.eval_flag:
        #     img = self.projection(img_ori, evals=True)
        # else:
        #     img = self.projection(img_ori, evals=False)
        img = img_ori
        h_list = []
        h_list.append(torch.mean(img, 0, True))
        # layer1 = F.leaky_relu(F.linear(img, self.weight1))
        layer1 = torch.relu(F.linear(img, self.weight1))
        # layer1 = F.linear(img, self.weight1)
        # layer1 = torch.relu(F.linear(img, self.weight1, self.bias1))
        # h_list.append(layer1)
        layer1 = self.dropout(layer1)
        # h_list.append(torch.mean(layer1, 0, True))
        h_list.append(layer1)
        layer2 = torch.relu(F.linear(layer1, self.weight2))
        # # layer1 = torch.relu(F.linear(img, self.weight1, self.bias1))
        layer2 = self.dropout(layer2)
        h_list.append(torch.mean(layer2, 0, True))
        out = F.linear(layer1, self.weight3)
        # out = F.linear(layer1, self.weight2, self.bias2)

        return out[:,:self.out_num], out, h_list


    def forward_eval(self, img_ori, maintain_model):

        para_lambda = 0.1
        # img = self.projection(img_ori, evals=False)
        img = img_ori
        h_list = []
        h_list.append(torch.mean(img, 0, True))
        # layer1 = F.leaky_relu(F.linear(img, self.weight1 - para_lambda * maintain_model.weight1))
        layer1 = torch.relu(F.linear(img, self.weight1 - 1 * para_lambda * maintain_model.weight1))
        # layer1 = F.linear(img, self.weight1)
        # layer1 = torch.relu(F.linear(img, self.weight1, self.bias1))
        # layer1 = self.dropout(layer1)
        h_list.append(torch.mean(layer1, 0, True))
        layer2 = torch.relu(F.linear(layer1, self.weight2 - para_lambda * maintain_model.weight2))
        # # layer1 = torch.relu(F.linear(img, self.weight1, self.bias1))
        # layer2 = self.dropout(layer2)
        h_list.append(torch.mean(layer2, 0, True))
        out = F.linear(layer1, self.weight3 - 1 * para_lambda * maintain_model.weight3)
        # out = F.linear(layer1, self.weight2, self.bias2)

        return out[:,:self.out_num]

    def forward_eval_1(self, img_ori, maintain_model):

        para_lambda = 0.8
        # img = self.projection(img_ori, evals=False)
        img = img_ori
        h_list = []
        h_list.append(torch.mean(img, 0, True))
        if not self.share_model.ready:
            self.share_model = maintain_model
        # layer1 = F.leaky_relu(F.linear(img, self.weight1 - para_lambda * maintain_model.weight1))
        layer1 = torch.relu(F.linear(img, self.weight1 - 1 * para_lambda * self.share_model.weight1))
        # layer1 = F.linear(img, self.weight1)
        # layer1 = torch.relu(F.linear(img, self.weight1, self.bias1))
        # layer1 = self.dropout(layer1)
        h_list.append(torch.mean(layer1, 0, True))
        layer2 = torch.relu(F.linear(layer1, self.weight2 - para_lambda * self.share_model.weight2))
        # # layer1 = torch.relu(F.linear(img, self.weight1, self.bias1))
        # layer2 = self.dropout(layer2)
        h_list.append(torch.mean(layer2, 0, True))
        out = F.linear(layer1, self.weight3 - 1 * para_lambda * self.share_model.weight3)
        # out = F.linear(layer1, self.weight2, self.bias2)

        return out[:,:self.out_num]

    def forward_eval_2(self, img_ori, maintain_model):
        # self.learn_p(img_ori)
        para_lambda = 0.1
        # img = self.projection(img_ori, evals=False)
        img = img_ori
        layer1 = F.linear(img, self.weight1)
        # layer1 = torch.relu(F.linear(img, self.weight1))
        # layer1 = F.linear(img, self.weight1)
        # layer1 = torch.relu(F.linear(img, self.weight1, self.bias1))
        # layer1 = self.dropout(layer1)
        layer11 = torch.relu(layer1) - para_lambda * torch.relu(F.linear(img, maintain_model.weight1))

        out = F.linear(layer11, self.weight3) - para_lambda * F.linear(layer11, maintain_model.weight3)
        # out = F.linear(layer1, self.weight2, self.bias2)

        return out[:,:self.out_num]

    def forward_eval_21(self, img_ori, maintain_model):
        # self.learn_p(img_ori)
        para_lambda = 0.1
        # img = self.projection(img_ori, evals=False)
        img = img_ori
        if not self.share_model.ready:
            self.share_model = maintain_model
        layer1 = F.linear(img, self.weight1)
        # layer1 = torch.relu(F.linear(img, self.weight1))
        # layer1 = F.linear(img, self.weight1)
        # layer1 = torch.relu(F.linear(img, self.weight1, self.bias1))
        # layer1 = self.dropout(layer1)
        layer11 = torch.relu(layer1) - 1 * para_lambda * torch.relu(F.linear(img, self.share_model.weight1))

        out = F.linear(layer11, self.weight3) - 0 * para_lambda * F.linear(layer11, self.share_model.weight3)
        # out = F.linear(layer1, self.weight2, self.bias2)

        return out[:,:self.out_num]


    def forward_eval_3(self, img_ori, maintain_model):
        # self.learn_p(img_ori)
        para_lambda = 0.1
        # img = self.projection(img_ori, evals=False)
        img = img_ori
        layer1 = F.linear(img, self.weight1)
        # layer1 = torch.relu(F.linear(img, self.weight1))
        # layer1 = F.linear(img, self.weight1)
        # layer1 = torch.relu(F.linear(img, self.weight1, self.bias1))
        # layer1 = self.dropout(layer1)
        layer11 = torch.relu(layer1 - para_lambda * F.linear(img, maintain_model.weight1))

        out = F.linear(layer11, self.weight3) - para_lambda * F.linear(layer11, maintain_model.weight3)
        # out = F.linear(layer1, self.weight2, self.bias2)

        return out[:,:self.out_num]


    def forward_eval_mode(self, img_ori, maintain_model):

        para_lambda = 0.0001
        # img = self.projection(img_ori, evals=False)
        img = img_ori

        # layer1 = F.leaky_relu(F.linear(img, self.weight1 - para_lambda * maintain_model.weight1))
        layer1 = torch.relu(F.linear(img, self.weight1 - 1 * para_lambda * maintain_model.weight1))
        # layer1 = self.dropout(layer1)
        layer2 = torch.relu(F.linear(layer1, self.weight2 - para_lambda * maintain_model.weight2))
        # layer2 = self.dropout(layer2)
        out = F.linear(layer1, self.weight3 - 1 * para_lambda * maintain_model.weight3)
        # out = F.linear(layer1, self.weight2, self.bias2)

        return out[:,:self.out_num]

    def forward_eval_mode1(self, img_ori, maintain_model):

        para_lambda = 0.001
        # img = self.projection(img_ori, evals=False)
        img = img_ori
        if not self.share_model_mode.ready:
            self.share_model_mode = maintain_model
        # layer1 = F.leaky_relu(F.linear(img, self.weight1 - para_lambda * maintain_model.weight1))
        layer1 = torch.relu(F.linear(img, self.weight1 - 1 * para_lambda * self.share_model_mode.weight1))
        # layer1 = self.dropout(layer1)
        layer2 = torch.relu(F.linear(layer1, self.weight2 - para_lambda * self.share_model_mode.weight2))
        # layer2 = self.dropout(layer2)
        out = F.linear(layer1, self.weight3 - 1 * para_lambda * self.share_model_mode.weight3)
        # out = F.linear(layer1, self.weight2, self.bias2)

        return out[:,:self.out_num]

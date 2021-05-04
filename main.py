#Version 2.1

import argparse
import os
import numpy as np
import math
from train.train import Train
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.autograd as autograd
import collections

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import helper
import copy
from model import model_utils
# import random

from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb
import pt_imagenet.dmodels.cifar as models
import time

print('=' * 60)
start = int(round(time.time()*1000))
print('Start Time: ' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start/1000)))
print('=' * 60)

# pdb.set_trace()
# torch.manual_seed(123)
# print(torch.manual_seed)

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', type=int, default=100,
                    help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=100,
                    help='size of the batches')
parser.add_argument('--neg_batch_size', type=int, default=100,
                    help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5,
                    help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999,
                    help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=12,
                    help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100,
                    help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28,
                    help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1,
                    help='number of image channels')
parser.add_argument('--sample_interval', type=int,
                    default=400, help='interval betwen image samples')
parser.add_argument('--gpu', type=int, default=0, help='gpu no.')
parser.add_argument('--dropoutrate', type=float, default=0.05, help='gpu no.')
parser.add_argument('--sample_rate', type=float, default=0.5, help='gpu no.')
parser.add_argument('--dataset', type=str, default='mnist',
                    choices=['mnist', 'cifar10', 'cifar100','emnist', 'twentynews', 'dbpedia'], help='gpu no.')
parser.add_argument('--negdataset', type=str, default='imagenet',
                    choices=['mnist', 'cifar10', 'imagenet', 'emnist'], help='negative dataset')
opt = parser.parse_args()

if 'cifar10' in opt.dataset:
    opt.img_size = 640
elif opt.dataset == 'mnist' or opt.dataset == 'emnist':
    opt.img_size = 784
elif opt.dataset == 'twentynews':
    opt.img_size = 2000
elif opt.dataset == 'dbpedia':
    opt.img_size = 768

# pdb.set_trace()
if opt.dataset == 'emnist':
    num_class_d = 47
if opt.dataset == 'mnist':
    num_class_d = 10
if opt.dataset == 'cifar10':
    num_class_d = 10
if opt.dataset == 'cifar100':
    num_class_d = 100
if opt.dataset == 'twentynews':
    num_class_d = 20
if opt.dataset == 'dbpedia':
    num_class_d = 14

opt.num_class_d = num_class_d


dataseize = np.prod((opt.channels, opt.img_size, opt.img_size))
torch.cuda.set_device(opt.gpu)
opt.cuda = True if torch.cuda.is_available() else False

groupsize = 2

first = int(num_class_d / groupsize)
if num_class_d % groupsize == 0:
    opt.class_groups = [groupsize] * first
else:
    opt.class_groups = [groupsize] * first + [num_class_d - first * groupsize]
opt.task_bounds = np.array([0] + opt.class_groups).cumsum()
opt.sample_rate = 0.1
opt.alpha1 = 0.0
opt.zoom = 30.0
opt.gradient_clip = 1.0
opt.num_tasks = len(opt.class_groups)

Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor
TensorB = torch.cuda.ByteTensor if opt.cuda else torch.ByteTensor
TensorL = torch.cuda.LongTensor if opt.cuda else torch.LongTensor

# ----------
#  Training
# ----------
epoch = 0
i = 0
# cnn = False
eps = 1e-8
gamma = 2.0
beta = 100  # 10 for mnist
epoch_num = 0
all_precision = []
all_recall = []
all_f1 = []
all_acc = []
CRITIC_ITERS = 1
LAMBDA = 10.0
opt.LAMBDA = LAMBDA

alpha = 1e-5  # 0.00003  # 8e-5
beta = 1.0
beta1 = 1.0
counter = 0
# pdb.set_trace()
threshold = torch.tensor(1.0).cuda().detach()
# ####################################################################################################

CLDataset = helper.load_dataset(opt.dataset)
cl_dataset = CLDataset(opt)
cl_dataset.intialization(TaskorClass=False)
labels = cl_dataset.labels
num_class = cl_dataset.num_classes
opt.num_class = num_class

opt.data_size = [cl_dataset.get_size(label_index) for label_index in range(opt.num_class)]
opt.cl_dataset = cl_dataset
train_function = Train(opt)
opt.total_data_train = np.sum(np.array(opt.data_size))
opt.test_set = cl_dataset.test_size
print(opt)

################################################################
#training classes
flag_test = False
for label_index in range(opt.num_class):
    epoch_num = 0
    flag_test = False
    # pdb.set_trace()
    i = 0

    train_function.global_index = label_index
    max_acc = 0

    if label_index > 0:
        train_function.set_model_parameters()

    for imgs in range(100000000):

        
        loss, finished_epoch, constraints, loss_penalty_n, loss_penalty_pn, loss1 = train_function.train_classes(label_index, i)
        
        # pdb.set_trace()
        if finished_epoch == -1:
            train_function.loadModel()
            flag_test = True
            break

            
        #############################################################################
        # Begin testing the learned model.
        if finished_epoch != epoch:

            epoch = finished_epoch
            i = 0
            # pdb.set_trace()
            if (epoch_num + 1) % (opt.max_epochs / 10) == 0:
            # if True:
                correct = 0.0
                correct_nodifference_0 = 0.0
                correct_nonornalization = 0.0
                correct_nonornalizationw = 0.0
                correct_nodifference_1 = 0.0
                all = 0.0

                train_function.eval()
                # for label_test in range(num_class):
                #     train_function.modelMain[label_test].eval()
                #     train_function.modelMain[label_test].eval_flag = True

                testiterator = cl_dataset.get_testing_iterator_usual()
                # i = 0
                prior_all = 0

                # print(epoch_num)
                # if epoch_num > opt.max_epochs * 2:
                #     # pdb.set_trace()
                #     modelMain[label_index].eval()
                #     score_temp_vm = []
                #     for index_s in range(80):
                #         positive_data = train_positive_iterater.get_next_batch(opt.neg_batch_size)
                #         positive_data_ = Tensor(positive_data)
                #         # pdb.set_trace()
                #         # positive_data_f_ = model_base.forward(positive_data_).data.view(-1, 640)
                #         score_temp = modelMain[label_index].forward(positive_data_)
                #         score_temp_vm.append(score_temp.data)

                #     score_temp_vm = torch.cat(score_temp_vm, dim=1)
                #     prior[label_index] = prior[label_index] * 0.5 + 0.5 * torch.mean(score_temp_vm[:,0] - score_temp_vm[:,1])
                #     prior_v[label_index] = prior_v[label_index] * 0.5 + 0.5 * torch.sqrt(torch.mean((score_temp_vm[:,0] - score_temp_vm[:,1]- prior[label_index])**2))

                
                test_labels = []
                    # pdb.set_trace()
                final_score = []
                for test_imgs in testiterator:

                    score = []
                    
                    if opt.dataset == 'mnist' or opt.dataset == 'emnist':
                        test_imgs_data = Tensor(test_imgs[0])

                    else:
                        test_imgs_data = Tensor(test_imgs[0])

                    test_labels.append(Tensor(test_imgs[1]))

                    
                    for label_index_test in range(0, label_index + 1):

                        score_temp = train_function.forward(label_index_test, test_imgs_data)
                        score_temp = torch.sigmoid(score_temp)

                        score.append(score_temp)
                    
                    final_score.append(torch.cat(score, dim=1))
                
                
                test_labels_f = torch.cat(test_labels).data
                final_score_f = torch.cat(final_score).data
                
                results = torch.argmax(final_score_f, dim=1).float()

                if epoch_num == opt.max_epochs and i == 0:
                    # pdb.set_trace()
                    print("=========================")
                    print(test_labels_f.data.cpu()[:20])
                    print(results.data.cpu()[:20])
                    print("=========================")
                    print(final_score_f.data.cpu()[:20])
                    print("=========================")


                label_predicted = torch.Tensor.float(
                    (results == test_labels_f).data)

                correct = torch.sum(label_predicted)

                all = float(label_predicted.size(0))

                train_function.train()

                acc = correct / all
                all_acc.append(acc)
                print("[Epoch %d/%d/%d] [Loss1: %0.2f] [PenP loss: %0.2f] [Constraints: %0.4f] [PenN: %0.4f] [PenPN: %0.4f] [Acc: %0.4f]" %
                    (finished_epoch + 1, opt.max_epochs, label_index + 1, loss1, loss, constraints, loss_penalty_n, loss_penalty_pn, correct / all))
                if flag_test:
                    break
                if acc >= max_acc:
                    max_acc = acc
                    train_function.saveModel()

            epoch_num += 1

        i = i+1


##############################################
#training tasks
opt.max_epochs = 100
train_function.cl_dataset.opt.max_epochs = opt.max_epochs
train_function.initial_task()
ending = int(round(time.time()*1000))
print('Class Training End Time: ' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(ending/1000)))
print("+" * 60)
print("start training tasks!")
print("wait...")
flag_test = False
for label_index in range(opt.num_tasks):
    epoch_num = 0
    flag_test = False
    i = 0

    train_function.global_index = label_index
    max_acc = 0


    for imgs in range(100000000):

        
        loss, finished_epoch = train_function.train_tasks(label_index)
        
        # pdb.set_trace()
        if finished_epoch == -1:
            # train_function.loadModel()
            flag_test = True
            break

            
        #############################################################################
        # Begin testing the learned model.
        if finished_epoch != epoch:
        # if True:


            epoch = finished_epoch
            i = 0
            # finished_epoch = -1
            # pdb.set_trace()
            # if True:
            if (epoch_num + 1) % (10) == 0:
                # correct = 0.0
                all = 0.0

                train_function.eval()

                testiterator = cl_dataset.get_testing_iterator_usual()
                # i = 0
                prior_all = 0
                
                test_labels = []
                    # pdb.set_trace()
                final_score = []
                for test_imgs in testiterator:

                    score = []
                    
                    if opt.dataset == 'mnist' or opt.dataset == 'emnist':
                        test_imgs_data = Tensor(test_imgs[0])

                    else:
                        test_imgs_data = Tensor(test_imgs[0])

                    test_labels.append(Tensor(test_imgs[1]))

                    
                    for label_index_test in range(0, label_index + 1):

                        score_temp = train_function.forwardFinal(label_index_test, test_imgs_data)

                        score.append(score_temp)
                    
                    final_score.append(torch.cat(score, dim=1))
                
                
                test_labels_f = torch.cat(test_labels).data
                final_score_f = torch.cat(final_score).data
                
                results = torch.argmax(final_score_f, dim=1).float()

                if epoch_num == opt.max_epochs and i == 0:
                    # pdb.set_trace()
                    print("=========================")
                    print(test_labels_f.data.cpu()[:20])
                    print(results.data.cpu()[:20])
                    print("=========================")
                    print(final_score_f.data.cpu()[:20])
                    print("=========================")


                label_predicted = torch.Tensor.float(
                    (results == test_labels_f).data)

                correct = torch.sum(label_predicted)

                all = float(label_predicted.size(0))

                train_function.train()

                acc = correct / all
                all_acc.append(acc)
                print("[Epoch %d/%d/%d] [Loss1: %0.2f] [PenP loss: %0.2f] [Constraints: %0.4f] [PenN: %0.4f] [PenPN: %0.4f] [Acc: %0.4f]" %
                    (finished_epoch + 1, opt.max_epochs, label_index + 1, loss1, loss, constraints, loss_penalty_n, loss_penalty_pn, correct / all))
                if flag_test:
                    break
                if acc >= max_acc:
                    max_acc = acc
                    # train_function.saveModel()

            epoch_num += 1

        i = i+1

print('=' * 60)
print('Start Time: ' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start/1000)))
print('=' * 60)
end = int(round(time.time()*1000))
print('End Time: ' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(end/1000)))
print('=' * 60)

# result = {'acc': all_acc}
# torch.save(result, './mnist_our2.pkl')

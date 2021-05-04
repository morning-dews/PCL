import os
import tarfile
import numpy as np
import helper.cl_dataset as cl_dataset
import helper.file_helper as file_manager
import matplotlib.pyplot as plt
import queue
from threading import Thread
import pdb
import torch

from scipy import misc

__author__ = 'garrett_local'

data_path_ot = '/home/huwenp/Dataset/imagenet/small32/imagenetOthers/'
data_path_pl = '/home/huwenp/Dataset/imagenet/small32/imagenetPositiveBig/'
data_path_nl = '/home/huwenp/Dataset/imagenet/small32/imagenetNegativeBig/'
data_path_ns = '/home/huwenp/Dataset/imagenet/small32/imagenetNegativeSmall/'
data_path_s = '/home/huwenp/Dataset/imagenet/small32/imagenetSmall/'
data_path_ps = '/home/huwenp/Dataset/imagenet/small32/imagenetPositiveSmall/'
data_path = '/home/huwenp/Dataset/imagenet/small32/imagenetOri'

others = [1, 6, 7, 9, 11, 12, 13, 14, 16, 22, 23, 24, 28, 30, 34, 35, 37, 38, 39, 40, 44, 48, 52, 53, 54, 57, 58,
          59, 60, 61, 62, 65, 67, 70, 73, 74, 75, 76, 77, 78, 80, 81, 83, 85, 92, 96, 100, 101, 102, 103, 104, 108, 111, 120,
          121, 122, 129, 135, 137, 138, 142, 147, 153, 157, 159, 162, 163, 165, 166, 167, 169, 170, 175, 178, 182, 183,
          185, 186, 188, 190, 193, 194, 195, 199, 203, 205, 206, 209, 212, 213, 214, 215, 216, 217, 218, 219, 220, 222,
          223, 224, 225, 226, 227, 228, 229, 233, 234, 248, 251, 252, 253, 254, 255, 256, 258, 259, 260, 262, 263, 277,
          287, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
          312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333,
          334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355,
          356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377,
          378, 379, 380, 381, 382, 396, 397, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456,
          457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478,
          479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 502, 503,
          504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525,
          526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547,
          548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569,
          570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591,
          592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613,
          614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635,
          636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657,
          658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679,
          680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701,
          702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723,
          724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745,
          746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767,
          768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789,
          790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811,
          812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833,
          834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855,
          856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877,
          878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899,
          900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921,
          922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943,
          944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965,
          966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987,
          988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000]

positives_ = [ 1, 6, 7, 9, 11, 12, 13, 14, 16, 22, 23, 24, 28, 30, 34, 35, 37, 38, 39, 40, 44, 48, 52, 53, 54, 57, 58,
          59, 60, 61, 62, 65, 67, 70, 73, 74, 75, 76, 77, 78, 80, 81, 83, 85, 92, 96, 100, 101, 102, 103, 104, 108, 111, 120,
          121, 122, 129, 135, 137, 138, 142, 147, 153, 157, 159, 162, 163, 165, 166, 167, 169, 170, 175, 178, 182, 183,
          185, 186, 188, 190, 193, 194, 195, 199, 203, 205, 206, 209, 212, 213, 214, 215, 216, 217, 218, 219, 220, 222,
          223, 224, 225, 226, 227, 228, 229, 233, 234, 248, 251, 252, 253, 254, 255, 256, 258, 259, 260, 262, 263, 277,
          287, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
          312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333]

negatives_ = [334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355,
          356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377,
          378, 379, 380, 381, 382, 396, 397, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456,
          457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478,
          479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 502, 503,
          504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525,
          526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547,
          548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569,
          570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591,
          592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613,
          614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635,
          636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657,
          658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679,
          680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701,
          702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723,
          724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745,
          746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767,
          768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789,
          790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811,
          812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833,
          834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855,
          856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877,
          878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899,
          900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921,
          922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943,
          944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965,
          966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987,
          988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000]

negatives_small = [334, 335, 336, 337, 338, 339, 340]
positives_small = [205, 206, 209, 212, 213]

def _binarize(labels):
    return ((labels == 0) | (labels == 1) | (labels == 8) |  # (labels == 4) |
            (labels == 9)) * 1  # * 2 - 1 (labels == 4) |

def _select_data(labels):
    label = labels == others[0]
    len_ = len(others)
    for i in range(1, len_):
        label = label | (labels == others[i])

    return label 
 
def _select_pOn_data(labels, style):
    label = labels == style[0]
    len_ = len(style)
    for i in range(1, len_):
        label = label | (labels == style[i])

    return label 


def _prepare_imagenet_data_all(data_path_t): 
    # pdb.set_trace()
    train_x = []
    train_y = []
    for i in range(1, 11):
        file_path = os.path.join(data_path, 'train_data_batch_{0:d}'.format(i))
        data_dict = file_manager.unpickle(file_path)
        train_x.append(data_dict['data'])
        train_y.append(data_dict['labels'])
        # pdb.set_trace()
    train_x = np.concatenate(train_x) / 255.0
    train_y = np.concatenate(train_y)

    data_dict = file_manager.unpickle(
        os.path.join(data_path, 'val_data'))
    test_x = data_dict['data'] / 255.0
    test_y = np.array(data_dict['labels'])
    train_x = train_x.reshape((train_x.shape[0], 3, 32, 32))
    # .transpose([0, 2, 3, 1])
    test_x = test_x.reshape((test_x.shape[0], 3, 32, 32))
    # pdb.set_trace()

    return train_x, train_y, test_x, test_y

def _make_Positive_Nagetive_data(data_path_p, data_path_n, positives , negatives, max_num = 10000):
    train_x = []
    train_y = []
    for i in range(1, 11):
        file_path = os.path.join(data_path, 'train_data_batch_{0:d}'.format(i))
        data_dict = file_manager.unpickle(file_path)
        train_x.append(data_dict['data'])
        train_y.append(data_dict['labels'])
    train_x = np.concatenate(train_x) / 255.0
    train_y = np.concatenate(train_y)

    # pdb.set_trace()

    label = _select_pOn_data(train_y, positives)
    train_x_ = train_x[label,:]
    for i in range(1000):
        lengh = train_x_.shape[0]
        if lengh > max_num:
            torch.save(train_x_[:max_num, :], data_path_p + 
                        'imagenet_batch_' + str(i) + '.pt')
            train_x_ = train_x_[max_num:, :]
        else:
            torch.save(train_x_[:lengh, :], data_path_p +
                       'imagenet_batch_' + str(i) + '.pt')
            train_x_ = None
        if train_x_ is None:
            break

    label = _select_pOn_data(train_y, negatives)
    train_x_ = train_x[label,:]
    for i in range(1000):
        lengh = train_x_.shape[0]
        if lengh > max_num:
            torch.save(train_x_[:max_num, :], data_path_n + 
                        'imagenet_batch_' + str(i) + '.pt')
            train_x_ = train_x_[max_num:, :]
        else:
            torch.save(train_x_[:lengh, :], data_path_n +
                       'imagenet_batch_' + str(i) + '.pt')
            train_x_ = None
        if train_x_ is None:
            break


def _make_Positive_Negative_small_data():
    train_x = []
    train_y = []
    for i in range(1, 11):
        file_path = os.path.join(data_path, 'train_data_batch_{0:d}'.format(i))
        data_dict = file_manager.unpickle(file_path)
        train_x.append(data_dict['data'])
        train_y.append(data_dict['labels'])
    train_x = np.concatenate(train_x) / 255.0
    train_y = np.concatenate(train_y)

    label = _select_pOn_data(train_y, positives)
    train_x_ = train_x[label,:]
    max_num = 20000
    for i in range(10):
        lengh = train_x_.shape[0]
        if lengh > max_num:
            torch.save(train_x_[:max_num, :], data_path_p + 
                        'imagenet_batch_' + str(i) + '.pt')
            train_x_ = train_x_[max_num:, :]
        else:
            torch.save(train_x_[:lengh, :], data_path_p +
                       'imagenet_batch_' + str(i) + '.pt')
            train_x_ = None
        if train_x_ is None:
            break

    label = _select_pOn_data(train_y, negatives)
    train_x_ = train_x[label,:]
    max_num = 20000
    for i in range(10):
        lengh = train_x_.shape[0]
        if lengh > max_num:
            torch.save(train_x_[:max_num, :], data_path_n + 
                        'imagenet_batch_' + str(i) + '.pt')
            train_x_ = train_x_[max_num:, :]
        else:
            torch.save(train_x_[:lengh, :], data_path_n +
                       'imagenet_batch_' + str(i) + '.pt')
            train_x_ = None
        if train_x_ is None:
            break

def _make_Negative_small_data():
    train_x = []
    train_y = []
    for i in range(1, 11):
        file_path = os.path.join(data_path, 'train_data_batch_{0:d}'.format(i))
        data_dict = file_manager.unpickle(file_path)
        train_x.append(data_dict['data'])
        train_y.append(data_dict['labels'])
    train_x = np.concatenate(train_x) / 255.0
    train_y = np.concatenate(train_y)

    # pdb.set_trace()

    label = _select_pOn_data(train_y, negatives_small)
    train_x_ = train_x[label,:]
    max_num = 20000
    for i in range(10):
        lengh = train_x_.shape[0]
        if lengh > max_num:
            torch.save(train_x_[:max_num, :], data_path_s + 
                        'imagenet_batch_' + str(i) + '.pt')
            train_x_ = train_x_[max_num:, :]
        else:
            torch.save(train_x_[:lengh, :], data_path_s +
                       'imagenet_batch_' + str(i) + '.pt')
            train_x_ = None
        if train_x_ is None:
            break


def _make_other_data():

    # folder = os.path.join(data_path, 'imagenet_train')
    # if not os.path.isdir(data_path):
    #     file_manager.download(url, data_path)
    #     with tarfile.open(full_path) as f:
    #         f.extractall(path=data_path)
    train_x = []
    train_y = []
    for i in range(1, 11):
        file_path = os.path.join(data_path, 'train_data_batch_{0:d}'.format(i))
        data_dict = file_manager.unpickle(file_path)
        train_x.append(data_dict['data'])
        train_y.append(data_dict['labels'])
    train_x = np.concatenate(train_x) / 255.0
    train_y = np.concatenate(train_y)

    # pdb.set_trace()

    label = _select_data(train_y)
    train_x = train_x[label,:]
    train_y = train_y[label]
    max_num = 5000
    for i in range(100000000):
        lengh = train_x.shape[0]
        if lengh > max_num:
            data_save = {'data':train_x[:max_num, :], 'labels':train_y[:max_num]}
            torch.save(data_save, data_path_ot + 
                        'imagenet_batch_' + str(i) + '.pt')
            train_x = train_x[max_num:, :]
            train_y = train_y[max_num:]
        else:
            data_save = {'data':train_x, 'labels':train_y}
            torch.save(data_save, data_path_ot +
                       'imagenet_batch_' + str(i) + '.pt')
            train_x = None
        if train_x is None:
            break

    # pdb.set_trace()
    # .transpose([0, 2, 3, 1])
    # train_x = train_x.reshape((train_x.shape[0], 3, 32, 32))
    # .transpose([0, 2, 3, 1])
    # test_x = test_x.reshape((test_x.shape[0], 3, 32, 32))
    
    # for i in range(10):
    #     misc.imsave('./images/imagenet' +
    #                 str(train_y[i]) + '-' + str(i) + '.jpg', train_x[i].transpose(1, 2, 0))

    # pdb.set_trace()
    # train_y = _binarize(train_y)
    # test_y = _binarize(test_y)
    # return train_x, train_y, test_x, test_y

class ImagenetIterator(object):

    def __init__(self, opt, data_path = None):
        # _make_other_data()
        # pdb.set_trace()
        # _prepare_imagenet_data_all(data_path)
        # opt = vars(args[0])
        # _make_Positive_Negative_small_data()
        # _make_Positive_Nagetive_data(data_path_ps, data_path_ns, positives_small, negatives_small)
        # self._train_x, self._train_y, self._test_x, self._test_y = \
        #     _prepare_imagenet_data(args[1])
        if data_path is None:
            self.data_path_t = data_path_ot
        else:
            self.data_path_t = data_path
        
        self.finished_epoch = -1
        self.opt = opt
        self.batch_size = opt.neg_batch_size

        # self.queue_raw = queue.Queue(10000)
        self.queue_batch = queue.Queue(100)

        self.thread_raw = Thread(target=self._prepare_imagenet_data)
        self.thread_raw.daemon = True
        self.thread_raw.start()

        ### Start the Thread(s) for make the batch from record data
        # self._train_x, self._train_y, self._test_x, self._test_y = \
        #     _prepare_imagenet_data_all(args[1])
        # super(ImagenetDataset, self).__init__(*args, **kwargs)

    def _prepare_imagenet_data(self): 
        train_x = []
        train_y = []
        while True:
            self.finished_epoch = self.finished_epoch + 1
            for i in range(0, 100000):
                if not os.path.exists(self.data_path_t + 'imagenet_batch_' + str(i) + '.pt'):
                    # pdb.set_trace()
                    break
                # if os.path.getsize(self.data_path_t + 'imagenet_batch_' + str(i) + '.pt') < 10:
                #     print(i)
                #     # break
                #     continue
                # print(i)
                batch = torch.load(self.data_path_t + 'imagenet_batch_' + str(i) + '.pt')

                train_x.append(batch['data'])
                train_y.append(batch['labels'])
                # pdb.set_trace()
                train_x = np.concatenate(train_x)
                train_y = np.concatenate(train_y)
                # len_all = train_x.shape[0]
                
                while True:
                    if train_x.shape[0] > self.batch_size:
                        mini_batch_x = train_x[:self.batch_size, :]
                        mini_batch_y = train_y[:self.batch_size]
                        train_x = train_x[self.batch_size:, :]
                        train_y = train_y[self.batch_size:]
                        # mini_batch_x = mini_batch_x.reshape((-1, 3, 32, 32))
                        mini_batch = (mini_batch_x, mini_batch_y)
                        self.queue_batch.put(mini_batch)
                    else:
                        train_x = [train_x]
                        train_y = [train_y]
                        break

    def __next__(self):
        return self.queue_batch.get()
    
    def get_next_batch(self):
        return self.queue_batch.get()


    def set_batchsize(self, batch_size):
        self.batch_size = batch_size
        while True:
            data_p = self.get_next_batch()
            if data_p[0].shape[0] == batch_size:
                break

    
    def _prepare_imagenet_test_data(self): 
            data_dict = file_manager.unpickle(
                os.path.join(self.data_path, 'val_data'))
            test_x = data_dict['data'] / 255.0
            test_y = np.array(data_dict['labels'])
            # .transpose([0, 2, 3, 1])
            test_x = test_x.reshape((test_x.shape[0], 3, 32, 32))
        # pdb.set_trace()


    def _original_train_x(self):
        return self._train_x

    def _original_train_y(self):
        return self._train_y

    def _original_test_x(self):
        return self._test_x

    def _original_test_y(self):
        return self._test_y

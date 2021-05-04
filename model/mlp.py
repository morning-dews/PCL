import torch.nn as nn
import pdb
import numpy as np
import torch


class Recognizer_MLP(nn.Module):
    def __init__(self, opt):
        super(Recognizer_MLP, self).__init__()
        # dense1_bn = nn.BatchNorm1d(512)
        # dense2_bn = nn.BatchNorm1d(256)
        self.shape = [2256, 2256]
        self.opt = opt
        self.img_shape = (opt.channels, opt.img_size, opt.img_size)
        self.rate = 0.8
        self.mask = [
            (torch.rand(self.shape[0], int(np.prod(self.img_shape))
                        ) > self.rate).float(),
            # (torch.rand(self.shape[0]) > self.rate).float(),
            (torch.rand(self.shape[1], self.shape[0]) > self.rate).float(),
            # (torch.rand(self.shape[1]) > self.rate).float(),
            # (torch.rand(512, 512) > self.rate).float(),
            # (torch.rand(512, 256) > self.rate).float(),
            torch.randint(1, 2, (1, self.shape[0]))
        ]
        self.false = True
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)),
                      self.shape[0], bias=self.false),
            nn.ReLU(),
            # nn.BatchNorm1d(self.shape[0]),
            nn.Dropout(self.opt.dropoutrate),
            ###################################
            nn.Linear(self.shape[0], self.shape[1], bias=self.false),
            nn.ReLU(),
            # nn.BatchNorm1d(self.shape[1]),
            nn.Dropout(self.opt.dropoutrate),
            # nn.Linear(512, 512, bias=False),
            # nn.ReLU(),
            # # nn.BatchNorm1d(512),
            # nn.Dropout(self.opt.dropoutrate),
            ###################################
            # nn.Linear(512, 256, bias=False),
            # nn.ReLU(),
            # # nn.BatchNorm1d(256),
            # nn.Dropout(self.opt.dropoutrate),
            nn.Linear(self.shape[1], 1, bias=self.false),
            # nn.ReLU()
            # nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)

        return validity

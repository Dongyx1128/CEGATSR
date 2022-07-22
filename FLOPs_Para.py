from thop import profile
from thop import clever_format

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import scipy.io as sio
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchnet import meter
import utils
import json
from data import HSTrainingData
from data import HSTestData
from CEGATSR import *
from common import *
# loss
from loss import HybridLoss
# from loss import HyLapLoss
from metrics import quality_assessment

# device = torch.device("cpu")
cuda = 1
device = torch.device("cuda" if cuda else "cpu")
class CEGATSR(nn.Module):
    def __init__(self, n_subs=4, n_ovls=1, in_feats=31, n_blocks=6, out_feats=64, n_scale=8, res_scale=1, use_share=True, conv=default_conv):
        super(CEGATSR, self).__init__()
        kernel_size = 3
        self.shared = use_share
        act = nn.ReLU(True)

        # calculate the group number (the number of branch networks)
        self.G = math.ceil((in_feats - n_ovls) / (n_subs - n_ovls))
        # calculate group indices
        self.start_idx = []
        self.end_idx = []

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > in_feats:
                end_ind = in_feats
                sta_ind = in_feats - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        if self.shared:
            self.branch = GCN_CNN_Unit(n_subs, out_feats, up_scale=n_scale//2, use_tail=True, conv=default_conv)
            # self.branch = GCN_CNN_Unit(n_subs, out_feats, use_tail=True, conv=default_conv)
            # up_scale=n_scale//2 means that we upsample the LR input n_scale//2 at the branch network, and then conduct 2 times upsampleing at the global network
        else:
            self.branch = nn.ModuleList
            for i in range(self.G):
                self.branch.append(GCN_CNN_Unit(n_subs, out_feats, up_scale=n_scale//2, use_tail=True, conv=default_conv))
                # self.branch.append(GCN_CNN_Unit(n_subs, out_feats, use_tail=True, conv=default_conv))

        self.trunk = Spatial_Spectral_Unit(in_feats, out_feats, n_blocks, act, res_scale, up_scale=2, use_tail=False, conv=default_conv)
        self.skip_conv = conv(in_feats, out_feats, kernel_size)
        self.final = conv(out_feats, in_feats, kernel_size)
        self.sca = n_scale//2

    def forward(self, x, lms):
        b, c, h, w = x.shape

        # Initialize intermediate “result”, which is upsampled with n_scale//2 times
        y = torch.zeros(b, c, self.sca * h, self.sca * w).cuda()

        channel_counter = torch.zeros(c).cuda()

        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            if self.shared:
                xi = self.branch(xi)
            else:
                xi = self .branch[g](xi)

            y[:, sta_ind:end_ind, :, :] += xi
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1

        # intermediate “result” is averaged according to their spectral indices
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)

        y = self.trunk(y)
        y = y + self.skip_conv(lms)
        y = self.final(y)

        return y


if __name__ == "__main__":
    model = CEGATSR().to(device)
    # x = torch.randn(1, 8, 16, 16)
    # lms = torch.randn(1, 8, 64, 64)
    x = torch.randn(1, 31, 16, 16).to(device)
    lms = torch.randn(1, 31, 128, 128).to(device)
    flops, params = profile(model, inputs=(x, lms,))
    print(f"result_1: FLOPs {flops} Params {params}")
    flops, params = clever_format([flops, params], "%.3f")
    print(f"result_2: FLOPs {flops} Params {params}")
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')
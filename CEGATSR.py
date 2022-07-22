import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import pdb

class Pre_ProcessLayer_Graph(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, stride, bias = True):
        super(Pre_ProcessLayer_Graph, self).__init__()
        self.head = prosessing_conv(in_feats, out_feats, kernel_size, stride, bias=bias)

    def forward(self, x):
        x = self.head(x)
        # print("conv.shape:", x.shape)
        [B, C, H, W] = x.shape
        y = torch.reshape(x, [B, C, H*W])
        N = H*W
        y = y.permute(0,2,1).contiguous()                # [B,C,N]->[B,N,C]
        adj = torch.zeros(B, N, N).cuda()                # adj:[N, N], 1 or 0
        k = 9
        for b in range(B):
            dist = cdist(y[b,:,:].cpu().detach().numpy(), y[b,:,:].cpu().detach().numpy(), metric='euclidean')
            # dist = dist + sp.eye(dist.shape[0])
            dist = np.where(dist.argsort(1).argsort(1) <= 6, 1, 0)        # k=9 + itself, all = 10, the largest 10 number is 1, rest is 0. 
            dist = torch.from_numpy(dist).type(torch.FloatTensor)
            dist = torch.unsqueeze(dist, 0)
            adj[b,:,:] = dist
        # y = y.permute(0,2,1).contiguous()       # [B,N,C]->[B,C,N]
        return y, adj


class ProcessLayer_Graph(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, stride, bias = True):
        super(ProcessLayer_Graph, self).__init__()
        self.last = transpose_conv(in_feats, out_feats, kernel_size, stride, bias=bias)

    def forward(self, x):
        y = self.last(x)
        return y


class GCN_Unit(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCN_Unit, self).__init__()
        kernel_size = 3
        stride = 2
        n_heads = 2
        dropout = 0.6
        alpha = 0.2
        self.head = Pre_ProcessLayer_Graph(in_feats, out_feats, kernel_size, stride, bias=True)
        self.body = GAT(out_feats, out_feats, dropout, alpha, n_heads)
        # self.body = nn.Conv2d(out_feats, out_feats, kernel_size, stride=1, padding=kernel_size // 2, bias=True)
        self.last = ProcessLayer_Graph(out_feats, out_feats, kernel_size, stride, bias=True)

        self.Act = nn.ReLU()

    def forward(self, x):
        y, adj = self.head(x)       # y.shape = torch.Size([16, 64, 32]), adj.shape = torch.Size([16, 64, 64])
        y = self.body(y, adj)       # y.shape = torch.Size([16, 64, 32])
        # y = self.body(y)       # y.shape = torch.Size([16, 64, 32])
        y = y.permute(0,2,1).contiguous()           # [B,N,C]->[B,C,N]
        [B,C,N] = y.shape
        H = int(math.sqrt(N))
        W = int(math.sqrt(N))
        y = torch.reshape(y,[B,C,H,W])
        # print("reshape later:y.shape:", y.shape)     # torch.Size([16, 64, 8, 8])
        y = self.last(y)        # GCN branch channel is "out_feats".
        # print("transconv:y.shape:", y.shape)     # torch.Size([16, 64, 16, 16])
        # pdb.set_trace()
        return y


class CNN_Unit(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size=3):
        super(CNN_Unit, self).__init__()
        self.point_conv = nn.Conv2d(
            in_channels=in_feats,
            out_channels=out_feats,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.depth_conv = nn.Conv2d(
            in_channels=out_feats,
            out_channels=out_feats,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_feats
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_feats)

    def forward(self, x):
        # y = self.point_conv(self.BN(x)
        y = self.point_conv(x)
        y = self.Act1(y)
        y = self.depth_conv(y)
        y = self.Act2(y)

        y = self.point_conv(y)
        y = self.Act1(y)
        y = self.depth_conv(y)
        y = self.Act2(y)
        return y


class GCN_CNN_Unit(nn.Module):          # GCN_CNN_Unit
    def __init__(self, in_feats, out_feats, up_scale, use_tail=True, conv=default_conv):  # up_scale
        super(GCN_CNN_Unit, self).__init__()
        kernel_size = 3
        self.pre = conv(in_feats, out_feats, kernel_size)
        self.head = GCN_Unit(out_feats, out_feats)
        self.body = CNN_Unit(out_feats, out_feats)
        self.last = conv(out_feats, out_feats, kernel_size)
        self.upsample = Upsampler(conv, up_scale, out_feats)
        self.tail = True
        if use_tail:
            self.tail = conv(out_feats, in_feats, kernel_size)

    def forward(self, x):
        # print("unit in_feats:",x.shape)
        y = self.pre(x)
        GCN_result = self.head(y)
        # print("GCN_result.shape:",GCN_result.shape)         # torch.Size([16, 64, 16, 16])
        CNN_result = self.body(y)
        # print("CNN_result.shape:",CNN_result.shape)         # torch.Size([16, 64, 16, 16])
        # pdb.set_trace()
        # y = torch.cat([GCN_result, CNN_result], dim=1)
        y = GCN_result
        y = self.last(y)
        # print("channel compress:", y.shape)     # torch.Size([16, 64, 16, 16])
        y = self.upsample(y)
        # print("upscale:", y.shape)      # torch.Size([16, 16, 32, 32])
        if self.tail is not None:
            y = self.tail(y)
            # print("reconstruct:",y.shape)    # torch.Size([16, 4, 32, 32])cave
        # pdb.set_trace()
        return y


class SSB(nn.Module):                   # SSB
    def __init__(self, in_feats, kernel_size, act, res_scale, conv=default_conv):
        super(SSB, self).__init__()
        self.spa = SpatialResBlock(conv, in_feats, kernel_size, act=act, res_scale=res_scale)
        self.spc = SpectralAttentionResBlock(conv, in_feats, 1, act=act, res_scale=res_scale)

    def forward(self, x):
        return self.spc(self.spa(x))


class SSPN(nn.Module):
    def __init__(self, in_feats, n_blocks, act, res_scale):
        super(SSPN, self).__init__()

        kernel_size = 3
        m = []

        for i in range(n_blocks):
            m.append(SSB(in_feats, kernel_size, act=act, res_scale=res_scale))

        self.net = nn.Sequential(*m)

    def forward(self, x):
        res = self.net(x)
        res += x

        return res


class Spatial_Spectral_Unit(nn.Module): # Spatial_Spectral_Unit
    def __init__(self, in_feats, out_feats, n_blocks, act, res_scale, up_scale, use_tail=False, conv=default_conv):
        super(Spatial_Spectral_Unit, self).__init__()
        kernel_size = 3
        self.head = conv(in_feats, out_feats, kernel_size)
        self.body = SSPN(out_feats, n_blocks, act, res_scale)
        self.upsample = Upsampler(conv, up_scale, out_feats)
        self.tail = None

        if use_tail:
            self.tail = conv(out_feats, in_feats, kernel_size)

    def forward(self, x):
        y = self.head(x)
        y = self.body(y)
        y = self.upsample(y)
        if self.tail is not None:
            y = self.tail(y)

        return y


class CEGATSR(nn.Module):
    def __init__(self, n_subs, n_ovls, in_feats, n_blocks, out_feats, n_scale, res_scale, use_share=True, conv=default_conv):
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
            self.start_idx.append(sta_ind)      # [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]
            self.end_idx.append(end_ind)        # [8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80, 86, 92, 98, 104, 110, 116, 122, 128]

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
                print("xi.shape:", xi.shape)

            y[:, sta_ind:end_ind, :, :] += xi
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1

        # intermediate “result” is averaged according to their spectral indices
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        # pdb.set_trace()
        y = self.trunk(y)
        y = y + self.skip_conv(lms)
        y = self.final(y)

        return y
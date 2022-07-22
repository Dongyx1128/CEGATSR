import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):        # Conv2d input: [B，C，H，W]. W=((w-k+2p)//s)+1
    if dilation == 1:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size-1) // 2, bias=bias)
    elif dilation == 2:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=2, bias=bias, dilation=dilation)

    else:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=3, bias=bias, dilation=dilation)


def prosessing_conv(in_channels, out_channels, kernel_size, stride, bias=True):      # W=((w-k+2p)//s)+1. [C,H,W]->[C,H/s,W/s]: k-2p=s.
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=bias)
    # s=2,h=w=8;s=3,h=w=6

def transpose_conv(in_channels, out_channels, kernel_size, stride, bias=True):       # [C,H/s,W/s]->[C,H,W]
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1, output_padding=1, bias=bias)

    # output = (input-1)*stride + outputpadding - 2*padding + kernelsize
    # 2p-op=k-s
    # s=2,p=1,outp=1,h=w=16, 2*pading-outpadding=1
    # s=3,p=1,outp=0,h=w=16, 2*pading-outpadding=2


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    input: (B,N,C_in)
    output: (B,N,C_out)
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):     # in_features=out_features。
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # definite trainable parameter, W and a
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))            # build matrix, size is (input_channel, output_channel)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)    # xavier
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # leakyrelu
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        """
        h: [B,N, in_features]  in_features
        adj: graph adjacent  [N, N], 0 or 1
        """
        # [B_batch,N_nodes,C_channels]
        B, N, C = x.size()
        h = torch.matmul(x, self.W)                                                     # [B,N,C], [B, N, out_features]
        # print("h.shape:",h.shape)       # torch.Size([16, 36, 64])
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N*N, self.out_features), h.repeat(1, N, 1)], dim=2).view(-1, N, N, 2*self.out_features)  # [B, N, N, 2*out_features]
        # print("a_input.shape:",a_input.shape)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))                    # [B, N, N, 1] => [B, N, N], eij∈(0-1)
        zero_vec = -1e12 * torch.ones_like(e)                                           # -endless
        # print("adj is cuda:", adj.is_cuda)     # return false or true
        attention = torch.where(adj>0, e, zero_vec)                                     # [B, N, N]
        # if the element in adj > 0，there is a connection between the 2 nodes，e remains, on the contrary, set mask as negative endless
        attention = F.softmax(attention, dim=2)                                         # [B,N,N]
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, h)                                               # [B,N,N]*[B,N,out_features]-> [B,N,out_features], Conversion of node information
        # print("h_prime.shape:", h_prime.shape)  # torch.Size([16, 64, 64])

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, n_heads):
        """
        Dense version of GAT.
        n_heads: multi-head, concat
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # definite multi-head
        self.attentions = [GraphAttentionLayer(in_features, out_features, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(out_features * n_heads, out_features, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        # print("GAT：x.shape:",x.shape)       # torch.Size([16, 64, 128])
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=2)


class CALayer(nn.Module):               # channel attention mechanism
    def __init__(self, in_channels, reduction_rate=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_rate, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_rate, in_channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SpatialResBlock(nn.Module):              # spatial attention block
    def __init__(self, conv, in_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(SpatialResBlock, self).__init__()
        m = []
        for i in range(2):                      # Conv - ReLU - Conv
            m.append(conv(in_feats, in_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(in_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class SpectralAttentionResBlock(nn.Module):     # spectral attention block
    def __init__(self, conv, in_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(SpectralAttentionResBlock, self).__init__()
        m = []
        for i in range(2):                      # Conv - ReLU - Conv
            m.append(conv(in_feats, in_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(in_feats))
            if i == 0:
                m.append(act)

        m.append(CALayer(in_feats, 16))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, up_scale, in_feats, bn=False, act=False, bias=True):
        m = []
        if (up_scale & (up_scale - 1)) == 0:
            for _ in range(int(math.log(up_scale, 2))):
                m.append(conv(in_feats, 4 * in_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(in_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(in_feats))

        elif up_scale == 3:
            m.append(conv(in_feats, 9 * in_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(in_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(in_feats))

        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
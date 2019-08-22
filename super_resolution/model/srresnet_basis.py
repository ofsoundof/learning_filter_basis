"""
Author: Yawei Li
Date: 20/08/2019
Basis learning method applied to SRResNet for ICCV2019 paper.
"""

import torch.nn as nn
import torch
from model import common


def make_model(args, parent=False):
    return SRResNet_Basis(args)


class SRResNet_Basis(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SRResNet_Basis, self).__init__()

        basis_size = args.basis_size
        n_basis = args.n_basis
        share_basis = args.share_basis
        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        scale = args.scale[0]
        bn_every = args.bn_every

        kernel_size = 3
        act = nn.PReLU()

        head = [conv(args.n_colors, n_feats, kernel_size=9), act]
        body = [common.ResBlock_Basis(n_feats, kernel_size, basis_size, n_basis, share_basis, bn=True, act=act,
                                      bn_every=bn_every) for _ in range(n_resblock)]
        body.extend([conv(n_feats, n_feats, kernel_size), nn.BatchNorm2d(n_feats)])

        tail = [
            common.Upsampler(conv, scale, n_feats, act=act),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.head(x)
        f = self.body(x)
        x = self.tail(x + f)
        return x

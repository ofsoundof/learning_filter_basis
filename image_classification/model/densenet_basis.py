"""
Paper: Learning Filter Basis for Convolutional Neural Network Compression. ICCV 2019
"""

import math
import torch
import torch.nn as nn
from model import common
import torch.nn.functional as F


def make_model(args, parent=False):
    return DenseNet_Basis(args)


def basis_extraction_pre_hook(module, input):
    if module.inverse_index:
        module.basis_weight = module.basis[:, -module.basis_size:, :, :]
    else:
        module.basis_weight = module.basis[:, :module.basis_size, :, :]


class conv_basis(nn.Module):
    def __init__(self, basis, in_channels, n_basis, basis_size, kernel_size, stride=1, bias=True, inverse_index=False):
        super(conv_basis, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.basis_size = basis_size
        self.n_basis = n_basis
        self.group = in_channels // basis_size
        self.inverse_index = inverse_index

        self.basis = basis
        self.handle = self.register_forward_pre_hook(basis_extraction_pre_hook)
        self.basis_bias = nn.Parameter(torch.zeros(n_basis)) if bias else None

    def forward(self, x):
        if self.group == 1:
            x = F.conv2d(input=x, weight=self.basis_weight, bias=self.basis_bias, stride=self.stride, padding=self.kernel_size//2)
        else:
            x = torch.cat([F.conv2d(input=xi, weight=self.basis_weight, bias=self.basis_bias, stride=self.stride,
                                    padding=self.kernel_size//2)
                           for xi in torch.split(x, self.basis_size, dim=1)], dim=1)
        return x

    def __repr__(self):
        s = 'Conv_basis(in_channels={}, basis_size={}, group={}, n_basis={}, kernel_size={}, out_channel={})'.format(
            self.in_channels, self.basis_size, self.group, self.n_basis, self.kernel_size, self.group * self.n_basis)
        return s


class BasicBlock(nn.Module):
    def __init__(self, conv, basis, in_channels, out_channels, n_basis, basis_size, kernel_size1, kernel_size2, stride=1,
                 bias=True, norm=common.default_norm, act=nn.ReLU, inverse_index=False):
        super(BasicBlock, self).__init__()
        group = in_channels // basis_size
        modules = []
        if norm: modules.append(norm(in_channels))
        if act: modules.append(act())
        modules.append(conv_basis(basis, in_channels=in_channels, n_basis=n_basis, basis_size=basis_size,
                                  kernel_size=kernel_size1, stride=stride, bias=bias, inverse_index=inverse_index))

        if norm: modules.append(norm(group * n_basis))
        if act: modules.append(act())
        modules.append(conv(group * n_basis, out_channels, kernel_size=kernel_size2, bias=bias))
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)


# reference: torchvision
class Dense(nn.Module):
    def __init__(
            self, basis, in_channels, k, bias=True,
            conv=BasicBlock,
            norm=common.default_norm,
            act=common.default_act, args=None):
        super(Dense, self).__init__()

        module_list = []
        self.n_basis = args.n_basis
        self.n_group = args.n_group
        self.inverse_index = args.inverse_index
        self.basis_size = in_channels // self.n_group
        self.k_size1 = args.k_size1
        self.k_size2 = args.k_size2
        module_list.append(conv(common.default_conv, basis, in_channels=in_channels, out_channels=k,
                                n_basis=self.n_basis, basis_size=self.basis_size, kernel_size1=self.k_size1,
                                kernel_size2=self.k_size2, bias=bias, norm=norm, act=act, inverse_index=self.inverse_index))

        self.body = nn.Sequential(*module_list)

    def forward(self, x):
        out = self.body(x)
        out = torch.cat((x, out), 1)

        return out


class BottleNeck(nn.Module):
    def __init__(
            self, in_channels, k, kernel_size=3, bias=False,
            conv=common.default_conv,
            norm=common.default_norm,
            act=common.default_act):

        super(BottleNeck, self).__init__()

        module_list = []
        if norm: module_list.append(norm(in_channels))
        module_list.append(act())
        module_list.append(nn.Conv2d(in_channels, 4 * k, 1, bias=bias))
        if norm: module_list.append(norm(4 * k))
        module_list.append(act())
        module_list.append(conv(4 * k, k, kernel_size, bias=bias))

        self.body = nn.Sequential(*module_list)

    def forward(self, x):
        out = self.body(x)
        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    def __init__(
            self, in_channels, transition_group, out_channels, bias=False,
            norm=common.default_norm,
            act=common.default_act):

        super(Transition, self).__init__()
        module_list = []
        if norm: module_list.append(norm(in_channels))
        module_list.append(act())
        module_list.append(nn.Conv2d(in_channels, in_channels // transition_group, 1, bias=bias))
        module_list.append(nn.Conv2d(in_channels // transition_group, out_channels, 1, bias=bias))
        self.module = nn.Sequential(*module_list)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(self.module(x) + x)


class DenseNet_Basis(nn.Module):
    def __init__(self, args, conv3x3=common.default_conv, conv1x1=None):
        super(DenseNet_Basis, self).__init__()
        # from IPython import embed; embed()
        args = args[0]
        n_basis = args.n_basis
        n_group = args.n_group
        kernel_size = args.k_size1

        n_blocks = (args.depth - 4) // 3
        if args.bottleneck: n_blocks //= 2

        k = args.k
        c_in = 2 * k

        basis_size = k * (n_blocks * 3 + 1) // n_group
        transition_group = args.transition_group
        def _dense_block(basis, in_channels):
            module_list = []
            for _ in range(n_blocks):
                if args.bottleneck:
                    module_list.append(BottleNeck(in_channels, k, conv=conv3x3))
                else:
                    module_list.append(Dense(basis, in_channels, k, conv=BasicBlock, args=args))
                in_channels += k

            return nn.Sequential(*module_list)

        self.basis = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(n_basis, basis_size, kernel_size, kernel_size)))#.to(self.device))
        module_list = []
        module_list.append(conv3x3(args.n_colors, c_in, 3, bias=False))

        for i in range(3):
            module_list.append(_dense_block(self.basis, c_in))
            c_in += k * n_blocks
            if i < 2:
                c_out = int(math.floor(args.reduction * c_in))
                module_list.append(Transition(c_in, transition_group, c_out))
                c_in = c_out

        module_list.append(common.default_norm(c_in))
        module_list.append(common.default_act())
        module_list.append(nn.AvgPool2d(8))
        self.features = nn.Sequential(*module_list)

        if args.data_train == 'ImageNet':
            n_classes = 1000
        else:
            if args.data_train.find('CIFAR') >= 0:
                n_classes = int(args.data_train[5:])

        self.classifier = nn.Linear(c_in, n_classes)

        common.init_kaiming(self)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.squeeze())

        return x

"""
Author: Yawei Li
Date: 21/08/2019
The final basis learning method for ResNet.
There are two basis set in one ResBlock group.
The 18 convs within one ResBlock group share the same basis. So there are 3 basis sets in total.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model import common


def make_model(args, parent=False):
    return ResNet_Basis_Blockwise(args)


class conv_basis(nn.Module):
    def __init__(self, basis, in_channels, n_basis, kernel_size, basis_size, stride=1, bias=True):
        super(conv_basis, self).__init__()
        self.in_channels = in_channels
        self.n_basis = n_basis
        self.kernel_size = kernel_size
        self.stride = stride
        self.basis_size = basis_size
        self.n_basis = n_basis
        self.group = in_channels // basis_size
        self.basis_weight = basis
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
    def __init__(self, conv, basis, in_channels, out_channels, kernel_size, kernel_size2=1, stride=1,
                 bias=True, bn=True, act=True):
        super(BasicBlock, self).__init__()
        n_basis, basis_size = basis.size()[:2]
        group = in_channels // basis_size
        modules = [conv_basis(basis, in_channels=in_channels, n_basis=n_basis,
                              kernel_size=kernel_size, basis_size=basis_size, stride=stride, bias=bias)]
        # if bn: modules.append(nn.BatchNorm2d(group * n_basis))
        # if act : modules.append(nn.ReLU())
        # print('group {}, n_basis {}, out_channels {}, bias {}, kernel_size1 {}, kernel_size2 {}'.
        #       format(group, n_basis, out_channels, bias, kernel_size, kernel_size2))
        modules.append(conv(group * n_basis, out_channels, kernel_size=kernel_size2, bias=bias))
        #if bn: modules.append(nn.BatchNorm2d(out_channels))
        #if act: modules.append(nn.ReLU())
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(
        self, in_channels, planes, kernel_size, stride=1,
        conv3x3=common.default_conv,
        conv1x1=common.default_conv,
        norm=common.default_norm,
        act=common.default_act,
        downsample=None):

        super(ResBlock, self).__init__()
        m = [conv3x3(
            in_channels, planes, kernel_size, stride=stride, bias=False
        )]
        if norm: m.append(norm(planes))
        m.append(act())
        m.append(conv3x3(planes, planes, kernel_size, bias=False))
        if norm: m.append(norm(planes))

        self.body = nn.Sequential(*m)
        self.downsample = downsample
        self.act_out = act()

    def forward(self, x):
        out = self.body(x)
        if self.downsample is not None: x = self.downsample(x)
        # print('Out size {}; x size {}'.format(out.size(), x.size()))
        out += x
        out = self.act_out(out)

        return out

# reference: torchvision
class ResBlockDecom(nn.Module):
    def __init__(
        self, basis_l1, basis_l2, in_channels, planes, kernel_size, kernel_size2, stride=1,
        conv3x3=BasicBlock,
        conv1x1=common.default_conv,
        norm=common.default_norm,
        act=common.default_act,
        downsample=None):

        super(ResBlockDecom, self).__init__()

        m = [conv3x3(conv1x1, basis_l1, in_channels, planes, kernel_size, kernel_size2, stride=stride),
             nn.BatchNorm2d(planes), nn.ReLU(),
             conv3x3(conv1x1, basis_l2, planes, planes, kernel_size, kernel_size2),
             nn.BatchNorm2d(planes)]

        self.body = nn.Sequential(*m)
        self.downsample = downsample
        self.act_out = act()

    def forward(self, x):
        out = self.body(x)
        if self.downsample is not None: x = self.downsample(x)
        out += x
        out = self.act_out(out)

        return out

class BottleNeck(nn.Module):
    def __init__(
        self, in_channels, planes, kernel_size, stride=1,
        conv3x3=common.default_conv,
        conv1x1=common.default_conv,
        norm=common.default_norm,
        act=common.default_act,
        downsample=None):

        super(BottleNeck, self).__init__()
        m = [conv1x1(in_channels, planes, 1, bias=False)]
        if norm: m.append(norm(planes))
        m.append(act())
        m.append(conv3x3(planes, planes, kernel_size, stride=stride, bias=False))
        if norm: m.append(norm(planes))
        m.append(act())
        m.append(conv1x1(planes, 4 * planes, 1, bias=False))
        if norm: m.append(norm(4 * planes))

        self.body = nn.Sequential(*m)
        self.downsample = downsample
        self.act_out = act()

    def forward(self, x):
        out = self.body(x)
        if self.downsample is not None: x = self.downsample(x)
        out += x
        out = self.act_out(out)

        return out

class DownSampleA(nn.Module):
    def __init__(self):
        super(DownSampleA, self).__init__()

    def forward(self, x):
        # identity shortcut with 'padding zero'
        c = x.size(1) // 2
        pool = F.avg_pool2d(x, 2)
        out = F.pad(pool, (0, 0, 0, 0, c, c), 'constant', 0)

        return out

class DownSampleC(nn.Sequential):
    def __init__(
        self, in_channels, out_channels,
        stride=1, conv1x1=common.default_conv):

        m = [
            conv1x1(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        super(DownSampleC, self).__init__(*m)

class ResNet_Basis_Blockwise(nn.Module):
    def __init__(self, args, conv3x3=common.default_conv):
        super(ResNet_Basis_Blockwise, self).__init__()
        args = args[0]
        self.args = args
        m = []
        if args.data_train.find('CIFAR') >= 0:
            self.expansion = 1
            self.n_basis1 = args.n_basis1
            self.n_basis2 = args.n_basis2
            self.n_basis3 = args.n_basis3
            self.basis_size1 = args.basis_size1
            self.basis_size2 = args.basis_size2
            self.basis_size3 = args.basis_size3
            self.n_blocks = (args.depth - 2) // 6
            self.k_size = args.kernel_size
            self.in_channels = 16
            self.downsample_type = 'A'
            self.basis1 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.n_basis1, self.basis_size1, self.k_size, self.k_size)))
            self.basis2 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.n_basis2, self.basis_size2, self.k_size, self.k_size)))
            self.basis3 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.n_basis3, self.basis_size3, self.k_size, self.k_size)))
            self.block = ResBlockDecom

            n_classes = int(args.data_train[5:])
            kwargs = {'kernel_size': args.kernel_size, 'conv3x3': conv3x3}
            m.append(common.BasicBlock(args.n_colors, 16, **kwargs))

            kwargs['conv3x3'] = BasicBlock
            kwargs['kernel_size2'] = args.k_size2
            kwargs['n_basis'] = self.n_basis1
            kwargs['basis_size'] = self.basis_size1
            kwargs['basis_l1'] = self.basis1
            kwargs['basis_l2'] = self.basis1

            m.append(self.make_layer(16, self.n_blocks, **kwargs))
            kwargs['n_basis'] = self.n_basis2
            kwargs['basis_size'] = self.basis_size2
            kwargs['basis_l1'] = self.basis1
            kwargs['basis_l2'] = self.basis2

            m.append(self.make_layer(32, self.n_blocks, stride=2, **kwargs))
            kwargs['n_basis'] = self.n_basis3
            kwargs['basis_size'] = self.basis_size3
            kwargs['basis_l1'] = self.basis2
            kwargs['basis_l2'] = self.basis3
            m.append(self.make_layer(64, self.n_blocks, stride=2, **kwargs))
            m.append(nn.AvgPool2d(8))

            fc = nn.Linear(64 * self.expansion, n_classes)

        self.features= nn.Sequential(*m)
        self.classifier = fc

        # only if when it is child model
        if conv3x3 == common.default_conv:
            if args.pretrained == 'download' or args.extend == 'download':
                state = getattr(models, 'resnet{}'.format(args.depth))(pretrained=True)
            elif args.extend:
                state = torch.load(args.extend)
            else:
                common.init_kaiming(self)
                return

            source = state.state_dict()
            target = self.state_dict()
            for s, t in zip(source.keys(), target.keys()):
                target[t].copy_(source[s])

    def make_layer(
        self, planes, blocks, kernel_size, kernel_size2, basis_l1, basis_l2, n_basis=16, basis_size=16, stride=1,
        conv3x3=common.default_conv,
        conv1x1=common.default_conv,
        bias=False):
        # print('basis_size {}, n_basis {}'.format(basis_size, n_basis))
        out_channels = planes * self.expansion
        if stride != 1 or self.in_channels != out_channels:
            if self.downsample_type == 'A':
                downsample = DownSampleA()
            elif self.downsample_type == 'C':
                downsample = DownSampleC(
                    self.in_channels,
                    out_channels,
                    stride=stride,
                    conv1x1=conv1x1
                )
        else:
            downsample = None
        kwargs = {'conv3x3': conv3x3, 'conv1x1': conv1x1}

        m = [self.block(basis_l1, basis_l2,
            self.in_channels, planes, kernel_size, kernel_size2,
            stride=stride, downsample=downsample, **kwargs
        )]
        self.in_channels = out_channels

        for _ in range(blocks - 1):
            m.append(self.block(basis_l2, basis_l2,
                self.in_channels, planes, kernel_size, kernel_size2, **kwargs
            ))

        return nn.Sequential(*m)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.squeeze())

        return x



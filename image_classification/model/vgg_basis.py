"""
Paper: Learning Filter Basis for Convolutional Neural Network Compression. ICCV 2019
"""

import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model import common
import torch.nn.functional as F


def make_model(args, parent=False):
    return VGG_Basis(args)


class conv_basis(nn.Module):
    def __init__(self, in_channels, basis_size, n_basis, kernel_size, stride=1, bias=True):
        super(conv_basis, self).__init__()
        self.in_channels = in_channels
        self.n_basis = n_basis
        self.kernel_size = kernel_size
        self.basis_size = basis_size
        self.stride = stride
        self.group = in_channels // basis_size
        self.weight = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(n_basis, basis_size, kernel_size, kernel_size)))
        self.bias = nn.Parameter(torch.zeros(n_basis)) if bias else None

    def forward(self, x):
        if self.group == 1:
            x = F.conv2d(input=x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2)
        else:
            x = torch.cat([F.conv2d(input=xi, weight=self.weight, bias=self.bias, stride=self.stride,
                                    padding=self.kernel_size//2)
                           for xi in torch.split(x, self.basis_size, dim=1)], dim=1)
        return x

    def __repr__(self):
        s = 'Conv_basis(in_channels={}, basis_size={}, group={}, n_basis={}, kernel_size={}, out_channel={})'.format(
            self.in_channels, self.basis_size, self.group, self.n_basis, self.kernel_size, self.group * self.n_basis)
        return s


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_basis, basis_size, kernel_size,
                 stride=1, bias=True, conv=common.default_conv, norm=common.default_norm, act=common.default_act):
        super(BasicBlock, self).__init__()
        group = in_channels // basis_size
        modules = [conv_basis(in_channels, basis_size, n_basis, kernel_size, stride, bias)]
        if norm is not None: modules.append(norm(group * n_basis))
        if act is not None: modules.append(act())
        modules.append(conv(group * n_basis, out_channels, kernel_size=1, stride=stride, bias=bias))
        if norm is not None: modules.append(norm(out_channels))
        if act is not None: modules.append(act())
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)


# reference: torchvision
class VGG_Basis(nn.Module):
    def __init__(self, args, conv3x3=common.default_conv, conv1x1=None):
        super(VGG_Basis, self).__init__()

        # we use batch noramlization for VGG
        args = args[0]
        norm = common.default_norm
        bias = not args.no_bias
        n_basis = args.n_basis
        basis_size = args.basis_size

        configs = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
            'ef': [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M']
        }

        body_list = []
        in_channels = args.n_colors
        for i, v in enumerate(configs[args.vgg_type]):
            if v == 'M':
                body_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                t = 3 if args.vgg_decom_type == 'all' else 8
                if i <= t:
                    body_list.append(common.BasicBlock(in_channels, v, args.kernel_size, bias=bias,
                                                       conv3x3=conv3x3, norm=norm))
                else:
                    body_list.append(BasicBlock(in_channels, v, n_basis, basis_size, args.kernel_size, bias=bias,
                                                conv=conv3x3, norm=norm))
                in_channels = v

        # assert(args.data_train.find('CIFAR') >= 0)
        self.features = nn.Sequential(*body_list)
        if args.data_train.find('CIFAR') >= 0:
            n_classes = int(args.data_train[5:])
            self.classifier = nn.Linear(in_channels, n_classes)
        elif args.data_train == 'ImageNet':
            n_classes = 1000
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, n_classes),
            )

        if conv3x3 == common.default_conv:
            model_dir = os.path.join('..', 'models')
            os.makedirs(model_dir, exist_ok=True)
            if args.data_train.find('CIFAR') >= 0:
                if args.pretrained == 'download' or args.extend == 'download':
                    url = (
                        'https://cv.snu.ac.kr/'
                        'research/clustering_kernels/models/vgg16-89711a85.pt'
                    )

                    state = model_zoo.load_url(url, model_dir=model_dir)
                elif args.extend:
                    state = torch.load(args.extend)
                else:
                    common.init_vgg(self)
                    return
            elif args.data_train == 'ImageNet':
                if args.pretrained == 'download':
                    url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
                    state = model_zoo.load_url(url, model_dir=model_dir)
                else:
                    common.init_vgg(self)
                    return
            else:
                raise NotImplementedError('Unavailable dataset {}'.format(args.data_train))
            # from IPython import embed; embed()
            self.load_state_dict(state, False)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


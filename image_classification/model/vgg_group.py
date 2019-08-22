import os
import torch
import torch.nn as nn
import torch.utils.model_zoo
from model import common
import torch.nn.functional as F


def make_model(args, parent=False):
    return VGG_GROUP(args)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_size, kernel_size,
                 stride=1, bias=True, conv=common.default_conv, norm=common.default_norm, act=common.default_act):
        super(BasicBlock, self).__init__()
        groups = in_channels // group_size
        modules = [conv(in_channels, in_channels, kernel_size=kernel_size, stride=stride, bias=bias, groups=groups)]
        if norm is not None: modules.append(norm(in_channels))
        if act is not None: modules.append(act())
        modules.append(conv(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias))
        if norm is not None: modules.append(norm(out_channels))
        if act is not None: modules.append(act())
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)


# reference: torchvision
class VGG_GROUP(nn.Module):
    def __init__(self, args, conv3x3=common.default_conv, conv1x1=None):
        super(VGG_GROUP, self).__init__()
        args = args[0]
        # we use batch noramlization for VGG
        norm = common.default_norm
        bias = not args.no_bias
        group_size = args.group_size

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
                    body_list.append(BasicBlock(in_channels, v, group_size, args.kernel_size, bias=bias,
                                                conv=conv3x3, norm=norm))
                in_channels = v

        # for CIFAR10 and CIFAR100 only
        assert(args.data_train.find('CIFAR') >= 0)
        n_classes = int(args.data_train[5:])

        self.features = nn.Sequential(*body_list)
        self.classifier = nn.Linear(in_channels, n_classes)

        if conv3x3 == common.default_conv:
            if args.pretrained == 'download' or args.extend == 'download':
                url = (
                    'https://cv.snu.ac.kr/'
                    'research/clustering_kernels/models/vgg16-89711a85.pt'
                )
                model_dir = os.path.join('..', 'models')
                os.makedirs(model_dir, exist_ok=True)
                state = torch.utils.model_zoo.load_url(url, model_dir=model_dir)
            elif args.extend:
                state = torch.load(args.extend)
            else:
                common.init_vgg(self)
                return

            self.load_state_dict(state, strict=False)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.squeeze())
        # from IPython import embed; embed(); exit()
        return x

def loss_type(loss_para_type):
    if loss_para_type == 'L1':
        loss_fun = nn.L1Loss()
    elif loss_para_type == 'L2':
        loss_fun = nn.MSELoss()
    else:
        raise NotImplementedError
    return loss_fun

def form_weight(args, basis, coordinate):
    size_b = basis.size()
    size_c = coordinate.size()
    n_basis = size_b[0]
    basis_size = size_b[1]
    in_channel = size_c[1]
    out_channel = size_c[0]
    group = in_channel // basis_size
    kernel_size = 3
    basis = torch.transpose(torch.reshape(basis, [n_basis, -1]), 0, 1)
    coordinate = torch.transpose(torch.reshape(torch.squeeze(coordinate), [out_channel * group, n_basis]), 0, 1)
    weight = torch.mm(basis, coordinate)
    weight = torch.reshape(torch.transpose(weight, 0, 1), [out_channel, in_channel, kernel_size, kernel_size])
    # weight = torch.reshape(weight.permute(4, 3, 0, 1, 2),
    #                        [out_channel, in_channel, kernel_size, kernel_size])
    return weight

def loss_norm_difference(model, args, pretrain_state, para_loss_type='L2'):
    #from IPython import embed; embed(); exit()

    current_state = list(model.parameters())
    pre_keys = list(pretrain_state.keys())
    pre_values = [v for _, v in pretrain_state.items()]
    loss_fun = loss_type(para_loss_type)
    loss = 0
    norm = 0
    t = 3 if args.vgg_decom_type == 'all' else 7
    for b in range(t, 13):

        pretrain_weight = pre_values[(b-t) * 6 + 18]

        # basis = current_state[(b-3) * 5 + 12]
        # coordinate = current_state[(b-3) * 5 + 13]
        if args.vgg_decom_type == 'all':
            basis = current_state[(b-t) * 8 + 12]
            coordinate = current_state[(b-t) * 8 + 16]
        else:
            basis = current_state[(b-t) * 8 + 28]
            coordinate = current_state[(b-t) * 8 + 32]


        current_weight = form_weight(args, basis, coordinate)
        # from IPython import embed; embed(); exit()
        loss += loss_fun(pretrain_weight, current_weight)
        norm += torch.mean(basis ** 2)

    return loss, norm

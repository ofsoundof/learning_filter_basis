"""
Paper: Factorized Convolutional Neural Networks. ICCV Workshop.
This method factorizes a convolutional layer into a couple cheap convolutions. Referred to as "Factor".
"""

import torch.nn as nn
from model import common


def make_model(args, parent=False):
    return VGG(args)


# class conv_sic(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
#         super(conv_sic, self).__init__()
#         self.channel_change = not in_channels == out_channels
#         conv = [nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
#                           padding=kernel_size // 2, groups=in_channels, bias=bias),
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias)]
#         self.body = nn.Sequential(*conv)
#         self.act = nn.ReLU()
#
#     def forward(self, x):
#         res = self.body(x)
#         x = self.act(res) if self.channel_change else self.act(res + x)
#         return x
#
#
# class conv_factor(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
#         super(conv_factor, self).__init__()
#         body = [conv_sic(in_channels, in_channels, kernel_size, stride=stride, bias=bias),
#                 conv_sic(in_channels, out_channels, kernel_size, stride=stride, bias=bias)]
#         self.body = nn.Sequential(*body)
#
#     def forward(self, x):
#         return self.body(x)
#
#
# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, bias=True, conv=conv_factor, norm=common.default_norm, act=common.default_act):
#         super(BasicBlock, self).__init__()
#         modules = [conv(in_channels, out_channels, kernel_size, stride, bias)]
#         if norm is not None: modules.append(norm(out_channels))
#         self.conv = nn.Sequential(*modules)
#
#     def forward(self, x):
#         return self.conv(x)


class conv_sic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, norm=common.default_norm, act=common.default_act):
        super(conv_sic, self).__init__()
        self.channel_change = not in_channels == out_channels
        conv = [nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                          padding=kernel_size // 2, groups=in_channels, bias=bias),
                norm(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias),
                norm(out_channels)]
        # if norm is not None: conv.append(norm(out_channels))
        self.body = nn.Sequential(*conv)
        if act is not None: self.act = nn.ReLU()


    def forward(self, x):
        res = self.body(x)
        x = self.act(res) if self.channel_change else self.act(res + x)
        return x


class conv_factor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, norm=common.default_norm, act=common.default_act):
        super(conv_factor, self).__init__()
        body = [conv_sic(in_channels, in_channels, kernel_size, stride=stride, bias=bias, norm=norm, act=act),
                conv_sic(in_channels, out_channels, kernel_size, stride=stride, bias=bias, norm=norm, act=act)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, bias=True, conv=conv_factor, norm=common.default_norm, act=common.default_act):
        super(BasicBlock, self).__init__()
        modules = [conv(in_channels, out_channels, kernel_size, stride, bias, norm, act)]
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)


# reference: torchvision
class VGG(nn.Module):
    def __init__(self, args, conv3x3=conv_factor, conv1x1=None):
        super(VGG, self).__init__()
        args = args[0]
        # we use batch noramlization for VGG
        norm = common.default_norm
        bias = not args.no_bias

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
                if i <=t:
                    body_list.append(common.BasicBlock(in_channels, v, args.kernel_size, bias=bias,
                                                       conv3x3=common.default_conv, norm=norm))
                else:
                    body_list.append(BasicBlock(in_channels, v, args.kernel_size, bias=bias, conv=conv3x3, norm=norm))
                in_channels = v

        # for CIFAR10 and CIFAR100 only
        assert(args.data_train.find('CIFAR') >= 0)
        n_classes = int(args.data_train[5:])

        self.features = nn.Sequential(*body_list)
        self.classifier = nn.Linear(in_channels, n_classes)

        # if conv3x3 == common.default_conv:
        #     if args.pretrained == 'download' or args.extend == 'download':
        #         url = (
        #             'https://cv.snu.ac.kr/'
        #             'research/clustering_kernels/models/vgg16-89711a85.pt'
        #         )
        #         model_dir = os.path.join('..', 'models')
        #         os.makedirs(model_dir, exist_ok=True)
        #         state = torch.utils.model_zoo.load_url(url, model_dir=model_dir)
        #     elif args.extend:
        #         state = torch.load(args.extend)
        #     else:
        #         common.init_vgg(self)
        #         return
        #
        #     self.load_state_dict(state)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.squeeze())

        return x


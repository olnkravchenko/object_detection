import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from loss import CenternetTTFLoss


input_height = input_width = 256


class Backbone(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.block_num = 1
        self.alpha = alpha
        self.filters = np.array([64 * self.alpha, 128 * self.alpha, 256 * self.alpha, 512 * self.alpha, 512 * self.alpha]).astype('int')
        s = self.filters
        self.layer1 = self.conv_bn_relu( 3,   s[0], False)
        self.layer2 = self.conv_bn_relu(s[0], s[0], True)  # stride 2
        self.layer3 = self.conv_bn_relu(s[0], s[1], False)
        self.layer4 = self.conv_bn_relu(s[1], s[1], True) # stride 4
        self.layer5 = self.conv_bn_relu(s[1], s[2], False)
        self.layer6 = self.conv_bn_relu(s[2], s[2], False)
        self.layer7 = self.conv_bn_relu(s[2], s[2], True) # stride 8
        self.layer8 = self.conv_bn_relu(s[2], s[3], False)
        self.layer9 = self.conv_bn_relu(s[3], s[3], False)
        self.layer10 = self.conv_bn_relu(s[3], s[3], True) # stride 16
        self.layer11 = self.conv_bn_relu(s[4], s[4], False)
        self.layer12 = self.conv_bn_relu(s[4], s[4], False)
        self.layer13 = self.conv_bn_relu(s[4], s[4], True) # stride 32

    def conv_bn_relu(self, input_num, output_num, max_pool=False, kernel_size=3):
        block = OrderedDict()
        block["conv_" + str(self.block_num)] = nn.Conv2d(input_num, output_num, kernel_size=kernel_size, stride=1, padding=1)
        block["bn_" + str(self.block_num)] = nn.BatchNorm2d(output_num, eps=1e-3, momentum=0.01)
        block["relu_" + str(self.block_num)] = nn.ReLU()
        if max_pool:
            block["pool_" + str(self.block_num)] = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.block_num += 1
        return nn.Sequential(block)

    def forward(self, x):
        out = self.layer1(x)
        out_stride_2 = self.layer2(out)
        out = self.layer3(out_stride_2)
        out_stride_4 = self.layer4(out)
        out = self.layer5(out_stride_4)
        out = self.layer6(out)
        out_stride_8 = self.layer7(out)
        out = self.layer8(out_stride_8)
        out = self.layer9(out)
        out_stride_16 = self.layer10(out)
        out = self.layer11(out_stride_16)
        out = self.layer12(out)
        out_stride_32 = self.layer13(out)
        return out_stride_2, out_stride_4, out_stride_8, out_stride_16, out_stride_32


class Head(nn.Module):
    def __init__(self, backbone_output_filters, class_number=20):
        super().__init__()
        self.connection_num = 3
        self.class_number = class_number
        self.backbone_output_filters = backbone_output_filters
        self.filters = [128, 64, 32]
        head_filters = [self.backbone_output_filters[-1]] + self.filters

        for i, filter_num in enumerate(self.filters):
            name = f'head_{i+1}'
            setattr(self, name, self.conv_bn_relu(name, head_filters[i], head_filters[i+1]))
            # create connection with backbone
            if i < self.connection_num:
                name = f'after_{-2-i}'
                setattr(self, name, self.conv_bn_relu(name, self.backbone_output_filters[-2-i],self.filters[i], 1))

        self.before_hm = self.conv_bn_relu("before_hm", self.filters[-1], self.filters[-1])
        self.before_sizes = self.conv_bn_relu("before_sizes", self.filters[-1], self.filters[-1])

        self.hm = self.conv_bn_relu("hm", self.filters[-1], self.class_number, 3, "sigmoid")
        self.sizes =  self.conv_bn_relu("hm", self.filters[-1], 4, 3, None)

    def conv_bn_relu(self, name, input_num, output_num, kernel_size=3, activation="relu"):
        block = OrderedDict()
        padding = 1 if kernel_size==3 else 0
        block["conv_" + name] = nn.Conv2d(input_num, output_num, kernel_size=kernel_size, stride=1, padding=padding)
        block["bn_" + name] = nn.BatchNorm2d(output_num, eps=1e-3, momentum=0.01)
        if activation == "relu":
            block["relu_" + name] = nn.ReLU()
        elif activation == "sigmoid":
            block["sigmoid_" + name] = nn.Sigmoid()
        return nn.Sequential(block)

    def connect_with_backbone(self, *backbone_out):
        used_out = [backbone_out[-i-2] for i in range(self.connection_num)]
        x = backbone_out[-1]
        for i in range(len(self.filters)):
            x = getattr(self, 'head_{}'.format(i+1))(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i < self.connection_num:
                name = f'after_{-2-i}'
                x_ = getattr(self, name)(used_out[i])
                x = torch.add(x, x_)
        return x

    def forward(self, *backbone_out):
        self.last_shared_layer = self.connect_with_backbone(self, *backbone_out)
        x = self.before_hm(self.last_shared_layer)
        hm_out = self.hm(x)

        x = self.before_sizes(self.last_shared_layer)
        sizes_out = self.sizes(x)

        x = torch.cat((hm_out, sizes_out), dim=1)
        return x


class ModelBuilder(nn.Module):
    """
    To connect head with backbone
    """
    def __init__(self, alpha=1.0, class_number=20):
        super().__init__()
        self.class_number = class_number
        self.backbone = Backbone(alpha)
        self.head = Head(backbone_output_filters=self.backbone.filters, class_number=class_number)
        self.loss = CenternetTTFLoss(class_number, 4, input_height//4, input_width//4)
    def forward(self, x, gt=None):
        x = x / 0.5 - 1.0     # normalization
        out = self.backbone(x)
        pred = self.head(*out)

        if gt is None:
            return pred
        else:
            loss = self.loss(gt, pred)
            return loss
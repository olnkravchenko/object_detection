import torch.nn as nn

from losses.centernet_ttf import CenternetTTFLoss
from centernet_head import Head
from centernet_backbone import Backbone


# todo (AA): move it somewhere
input_height = input_width = 256

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

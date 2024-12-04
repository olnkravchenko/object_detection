from torch import nn

from core.backbones import Backbone
from core.heads import Head
from core.loss_functions import CenternetTTFLoss


class ModelBuilder(nn.Module):
    """
    To connect head with backbone
    """

    def __init__(self, width: int, height: int, alpha=1.0, class_number=20):
        super().__init__()
        self.class_number = class_number
        self.backbone = Backbone(alpha)
        self.head = Head(
            backbone_output_filters=self.backbone.filters,
            class_number=class_number,
        )
        self.loss = CenternetTTFLoss(class_number, 4, height // 4, width // 4)

    def forward(self, x, gt=None):
        x = x / 0.5 - 1.0  # normalization
        out = self.backbone(x)
        pred = self.head(*out)

        if gt is None:
            return pred
        else:
            loss = self.loss(gt, pred)
            return loss

from abc import ABC, abstractmethod

import numpy as np
from utils import gaussian as gaus


class Encoder(ABC):

    @abstractmethod
    def encode(self, bboxes, labels):
        pass


class CenternetEncoder(Encoder):

    def __init__(self, width=320, height=320, down_ratio=4, n_classes=20):
        self._img_width = width
        self._img_height = height
        self._out_width = width // down_ratio
        self._out_height = height // down_ratio
        self._n_classes = n_classes
        self._down_ratio = down_ratio
        print(f"down_ratio = {self._down_ratio}")

    def encode(self, bboxes, labels):
        heat_map = np.zeros(
            (self._out_height, self._out_width, self._n_classes),
            dtype=np.float32,
        )
        coors = np.zeros((self._out_height, self._out_width, 4), dtype=np.float32)
        for cls_id, bbox in zip(labels.data.numpy(), bboxes.data.numpy()):
            box_s = bbox / self._down_ratio
            h, w = box_s[3] - box_s[1], box_s[2] - box_s[0]
            rad_w_class = int(np.round(gaus.gaussian_radius([h, w])))
            rad_h_class = rad_w_class
            if h > 0 and w > 0:
                center = np.array(
                    [(box_s[0] + box_s[2]) / 2, (box_s[1] + box_s[3]) / 2],
                    dtype=np.float32,
                )
                center = np.round(center)
                center = np.clip(
                    center, [0, 0], [self._out_width - 1, self._out_height - 1]
                )
                center_int = center.astype(np.int32)
                gaus.draw_gaussian(
                    heat_map[..., cls_id - 1],
                    center_int,
                    rad_w_class,
                    rad_h_class,
                )
                coors[center_int[1], center_int[0]] = bbox
        return np.concatenate((heat_map, coors), axis=-1)

from typing import TypeVar

import torch
from datasets.encoders import Encoder
from torch.utils import data

TEncoder = TypeVar("Encoder", bound=Encoder)


class DatasetIterator(data.Dataset):

    def __init__(
        self, dataset: data.Dataset, transformer, encoder: TEncoder
    ) -> None:
        self._dataset = dataset
        self._transformation = transformation
        self._encoder = encoder

    def __getitem__(self, index):
        img, lbl = self._dataset[index]
        img_, bboxes_, labels_ = self._transformation(
            img, lbl["boxes"], lbl["labels"]
        )
        lbl_encoded = self._encoder(bboxes_, labels_)
        return img_, torch.from_numpy(lbl_encoded)

    def __len__(self):
        return len(self._dataset)

    def encode(self, width: int, height: int, encoder):
        self._dataset = encoder(width, height, self._dataset)

    def get_examples(self, batch_size: int = 10):
        return data.DataLoader(
            self._dataset, batch_size=batch_size, num_workers=2, shuffle=True
        )

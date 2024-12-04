from typing import TypeVar

import torch
from datasets.encoders import Encoder
from torch.utils import data

TEncoder = TypeVar("Encoder", bound=Encoder)
BATCH_LIMIT = 32


class DatasetIterator(data.Dataset):

    def __init__(self, dataset: data.Dataset, transformer, encoder: TEncoder) -> None:
        self._dataset = dataset
        self._transformer = transformer
        self._encoder = encoder

    def __getitem__(self, index):
        img, lbl = self._dataset[index % BATCH_LIMIT]
        img, bboxes, labels = self._transformer(img, lbl["boxes"], lbl["labels"])
        encoded_lbl = self._encoder.encode(bboxes, labels)
        return img, torch.from_numpy(encoded_lbl)

    def __len__(self):
        return BATCH_LIMIT or len(self._dataset)

    def get_examples(self, *, batch_size: int = 10):
        return data.DataLoader(
            self._dataset, batch_size=batch_size, num_workers=2, shuffle=True
        )

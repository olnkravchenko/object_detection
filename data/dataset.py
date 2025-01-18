import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, dataset_loader, transformation, encoder):
        self._dataset_loader = dataset_loader
        self._transformation = transformation
        self._encoder = encoder

    def __getitem__(self, index):
        img, lbl = self._dataset_loader[index]
        print(lbl)
        img_, bboxes_, labels_ = self._transformation(
            img, lbl.get("boxes", []), lbl.get("labels", [])
        )
        lbl_encoded = self._encoder(bboxes_, labels_)
        return img_, torch.from_numpy(lbl_encoded)

    def __len__(self):
        return len(self._dataset_loader)

import os
from abc import abstractmethod

from torchvision.datasets import CocoDetection, VisionDataset, VOCDetection


class DataLoader(VisionDataset):
    # TODO: add initialization from the VisionDataset
    def __init__(self, dataset_path: str, image_set: str = None):
        self.image_set = "train" if image_set is None else image_set
        self.dataset_path = dataset_path

    @abstractmethod
    def load(self):
        """
        Loads data and returns pytorch VisionDataset
        """


class PascalVOCDataLoader(DataLoader, VOCDetection):

    def load(self):
        """
        The dataset is automatically downloaded if `dataset_path` does not exist
        """

        is_download = not os.path.exists(self.dataset_path)
        return VOCDetection(
            root=self.dataset_path,
            year="2007",
            image_set=self.image_set,
            download=is_download,
        )


class CocoDataLoader(DataLoader):

    # TODO: implement
    def load(self):
        pass

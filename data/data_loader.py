import os
from abc import abstractmethod

from torchvision.datasets import CocoDetection, VisionDataset, VOCDetection
from utils.io_utils import download_file, unzip_archive


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


class MSCocoDataLoader(DataLoader):

    __urls = {
        "train": {
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "images": "http://images.cocodataset.org/zips/train2017.zip",
        },
        "val": {
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "images": "http://images.cocodataset.org/zips/val2017.zip",
        },
        "test": {
            "annotations": "http://images.cocodataset.org/annotations/image_info_test2017.zip",
            "images": "http://images.cocodataset.org/zips/test2017.zip",
        },
    }

    def load(self):
        if os.path.exists(self.dataset_path):
            # TODO: wrap into CocoDetection
            return

        urls = self.__urls[self.image_set].items()
        for name, url in urls:
            print(f"Downloading {name}...")
            download_file(url, self.dataset_path)
            unzip_archive(self.dataset_path, self.dataset_path)
            # TODO: wrap into CocoDetection


class CustomDataLoader(DataLoader):
    # TODO: implement, find corresponding Detection class
    def load(self):
        pass

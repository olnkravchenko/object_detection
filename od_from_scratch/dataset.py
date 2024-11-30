#dataset.py
import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class PascalVOCDataset(Dataset):
    def __init__(self, root_dir,
                 num_images=10,  # Кількість зображень для оверфіту
                 transform=None):
        """
        Args:
            root_dir (string): Директорія з зображеннями та анотаціями
            num_images (int): Кількість зображень для вибору
            transform (callable, optional): Трансформації зображень
        """
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # PASCAL VOC classes
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        # Підготовка шляхів зображень та анотацій
        self.image_paths = []
        self.annotation_paths = []

        # Вибираємо перші num_images зображень з валідними анотаціями
        jpg_files = [f for f in os.listdir(os.path.join(root_dir, 'JPEGImages')) if f.endswith('.jpg')]

        for filename in jpg_files[:num_images]:
            image_path = os.path.join(root_dir, 'JPEGImages', filename)
            annotation_path = os.path.join(root_dir, 'Annotations', filename.replace('.jpg', '.xml'))

            if os.path.exists(annotation_path):
                self.image_paths.append(image_path)
                self.annotation_paths.append(annotation_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Завантаження зображення
        image = Image.open(self.image_paths[idx]).convert('RGB')

        # Парсинг анотації
        tree = ET.parse(self.annotation_paths[idx])
        root = tree.getroot()

        # Збір боксів та міток
        boxes = []
        labels = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in self.classes:
                label = self.classes.index(class_name) + 1  # Індексація з 1

                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

        # Перетворення на тензори
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # Застосування трансформацій
        if self.transform:
            image = self.transform(image)

        return image, boxes, labels


def collate_fn(batch):
    """
    Власний колейт-фн для обробки боксів змінного розміру
    """
    images = [item[0] for item in batch]
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    images = torch.stack(images, 0)

    return images, boxes, labels


# Функція для створення датасету для оверфіту
def get_overfitting_dataset(root_dir, num_images=5):
    """
    Повертає невеликий набір зображень для цільового оверфіту
    """
    return PascalVOCDataset(
        root_dir=root_dir,
        num_images=num_images
    )
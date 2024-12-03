import os

import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Subset
import torchvision.transforms.v2 as transforms
from collections import OrderedDict
from torchsummary import summary
from matplotlib import pyplot as plt

# Importing custom modules (ensure these are in the same directory)
from loss import CenternetTTFLoss
from encoder import CenternetEncoder
from visualizer import get_image_with_bboxes

print("GPU is available: ", torch.cuda.is_available() )

input_height = input_width = 256
down_ratio = 4


def load_model_weights(model, checkpoint_path, continue_training=False):
    """
    Load model weights from a checkpoint file.

    Args:
        model (ModelBuilder): The model to load weights into
        checkpoint_path (str): Path to the model checkpoint file
        continue_training (bool):
            - If True: Load weights and continue training from the last checkpoint
            - If False: Start a new training process from scratch

    Returns:
        ModelBuilder: Model with loaded weights
        int: Starting epoch number (0 if not continuing training)
    """
    if continue_training and os.path.exists(checkpoint_path):
        try:
            # Load the state dictionary
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint)

            # Extract epoch number from filename (assuming format: pascal_voc_epoch_{epoch}_weights.pt)
            start_epoch = int(os.path.splitext(os.path.basename(checkpoint_path))[0].split('_')[-1])

            print(f"Continuing training from checkpoint: {checkpoint_path}")
            print(f"Starting from epoch: {start_epoch}")

            return model, start_epoch

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")

    # If continue_training is False or checkpoint doesn't exist
    print("Starting new training process")
    return model, 0


def main():
    # Check GPU availability
    print("GPU is available: ", torch.cuda.is_available())

    # Load and prepare dataset
    dataset_val = torchvision.datasets.VOCDetection(root="VOC", year='2007', image_set="val", download=True)
    dataset_val = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset_val)

    # Select first 30 indices
    indices = list(range(30))
    dataset_val = Subset(dataset_val, indices)
    print(len(dataset_val))

    # Initialize encoder and transform
    encoder = CenternetEncoder(input_height, input_width)
    transform = transforms.Compose([
        transforms.Resize(size=(input_width, input_height)),
        transforms.ToTensor()
    ])

    # Create custom dataset
    torch_dataset = Dataset(
        dataset=dataset_val,
        transformation=transform,
        encoder=encoder
    )

    # Create batch generator
    batch_generator = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=5,
        num_workers=0,
        shuffle=False
    )

    # Model and training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelBuilder(alpha=0.25).to(device)
    summary(model, input_size=(3, 256, 256), batch_size=-1)

    # Decide whether to continue training or start new
    CONTINUE_TRAINING = True  # Set to False to start a new training process
    CHECKPOINT_PATH = 'model_checkpoints/pascal_voc_final_weights.pt'

    # Load weights if continuing training
    model, start_epoch = load_model_weights(model, CHECKPOINT_PATH, CONTINUE_TRAINING)

    # Optimizer
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training
    EPOCHS = 1
    steps_per_epoch = 50

    os.makedirs('model_checkpoints', exist_ok=True)

    for epoch in range(start_epoch, EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))
        model.train()

        for step in range(steps_per_epoch):
            # Cycle through the dataset
            input, gt_data = next(iter(batch_generator))

            input = input.to(device).contiguous()
            gt_data = gt_data.to(device)
            gt_data.requires_grad = False

            loss_dict = model(input, gt=gt_data)
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()

            print(loss_dict['loss'])

        torch.save(model.state_dict(), f'model_checkpoints/pascal_voc_epoch_{epoch + 1}_weights.pt')
        print(f'Weights saved for epoch {epoch + 1}')

    torch.save(model.state_dict(), 'model_checkpoints/pascal_voc_final_weights.pt')
    print('Final weights saved')

def main():
    # Check GPU availability
    print("GPU is available: ", torch.cuda.is_available())

    # Load and prepare dataset
    dataset_val = torchvision.datasets.VOCDetection(root="VOC", year='2007', image_set="val", download=True)
    dataset_val = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset_val)

    # Select first 30 indices
    indices = list(range(30))
    dataset_val = Subset(dataset_val, indices)
    print(len(dataset_val))

    # Initialize encoder and transform
    encoder = CenternetEncoder(input_height, input_width)
    transform = transforms.Compose([
        transforms.Resize(size=(input_width, input_height)),
        transforms.ToTensor()
    ])

    # Create custom dataset
    torch_dataset = Dataset(
        dataset=dataset_val,
        transformation=transform,
        encoder=encoder
    )

    # Create batch generator
    batch_generator = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=5,
        num_workers=0,
        shuffle=False
    )

    # Model and training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelBuilder(alpha=0.25).to(device)
    summary(model, input_size=(3, 256, 256), batch_size=-1)

    CONTINUE_TRAINING = True  # Set to False to start a new training process
    CHECKPOINT_PATH = 'model_checkpoints/pascal_voc_final_weights.pt'

    # Load weights if continuing training
    model, start_epoch = load_model_weights(model, CHECKPOINT_PATH, CONTINUE_TRAINING)

    # Optimizer
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training
    EPOCHS = 1
    steps_per_epoch = 50

    os.makedirs('model_checkpoints', exist_ok=True)

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))
        model.train()

        for step in range(steps_per_epoch):
            # Cycle through the dataset
            input, gt_data = next(iter(batch_generator))

            input = input.to(device).contiguous()
            gt_data = gt_data.to(device)
            gt_data.requires_grad = False

            loss_dict = model(input, gt=gt_data)
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()

            print(loss_dict['loss'])
        torch.save(model.state_dict(), f'model_checkpoints/pascal_voc_epoch_{epoch + 1}_weights.pt')
        print(f'Weights saved for epoch {epoch + 1}')
    torch.save(model.state_dict(), 'model_checkpoints/pascal_voc_final_weights.pt')
    print('Final weights saved')

class Dataset(data.Dataset):
    def __init__(self, dataset, transformation, encoder):
        self._dataset = dataset
        self._transformation = transformation
        self._encoder = encoder

    def __getitem__(self, index):
        img, lbl = self._dataset[index]
        img_, bboxes_, labels_ = self._transformation(img, lbl['boxes'], lbl['labels'])
        lbl_encoded = self._encoder(bboxes_, labels_)
        return img_, torch.from_numpy(lbl_encoded)

    def __len__(self):
        return len(self._dataset)

class Backbone(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.block_num = 1
        self.alpha = alpha
        self.filters = np.array([64 * self.alpha, 128 * self.alpha, 256 * self.alpha, 512 * self.alpha, 512 * self.alpha]).astype('int')
        s = self.filters

        self.layer1 = self.conv_bn_relu(3, s[0], False)
        self.layer2 = self.conv_bn_relu(s[0], s[0], True)  # stride 2
        self.layer3 = self.conv_bn_relu(s[0], s[1], False)
        self.layer4 = self.conv_bn_relu(s[1], s[1], True)  # stride 4
        self.layer5 = self.conv_bn_relu(s[1], s[2], False)
        self.layer6 = self.conv_bn_relu(s[2], s[2], False)
        self.layer7 = self.conv_bn_relu(s[2], s[2], True)  # stride 8
        self.layer8 = self.conv_bn_relu(s[2], s[3], False)
        self.layer9 = self.conv_bn_relu(s[3], s[3], False)
        self.layer10 = self.conv_bn_relu(s[3], s[3], True)  # stride 16
        self.layer11 = self.conv_bn_relu(s[4], s[4], False)
        self.layer12 = self.conv_bn_relu(s[4], s[4], False)
        self.layer13 = self.conv_bn_relu(s[4], s[4], True)  # stride 32

    def conv_bn_relu(self, input_num, output_num, max_pool=False, kernel_size=3):
        block = OrderedDict()
        block[f"conv_{self.block_num}"] = nn.Conv2d(input_num, output_num, kernel_size=kernel_size, stride=1, padding=1)
        block[f"bn_{self.block_num}"] = nn.BatchNorm2d(output_num, eps=1e-3, momentum=0.01)
        block[f"relu_{self.block_num}"] = nn.ReLU()
        if max_pool:
            block[f"pool_{self.block_num}"] = nn.MaxPool2d(kernel_size=2, stride=2)
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
    def __init__(self, alpha=1.0, class_number=20):
        super().__init__()
        self.class_number = class_number
        self.backbone = Backbone(alpha)
        self.head = Head(backbone_output_filters=self.backbone.filters, class_number=class_number)
        self.loss = CenternetTTFLoss(class_number, 4, input_height//4, input_width//4)

    def forward(self, x, gt=None):
        x = x / 0.5 - 1.0  # normalization
        out = self.backbone(x)
        pred = self.head(*out)

        if gt is None:
            return pred
        else:
            loss = self.loss(gt, pred)
            return loss

if __name__ == "__main__":
    main()
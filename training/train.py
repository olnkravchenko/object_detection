import argparse
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils import data

from data.dataset import Dataset
from models.centernet import ModelBuilder, input_height, input_width
from training.encoder import CenternetEncoder

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--overfit', action='store_true', help='overfit to 10 images')
args = parser.parse_args()

overfit = args.overfit

dataset_val = torchvision.datasets.VOCDetection(
    root="../VOC", year="2007", image_set="val", download=False
)

transform = transforms.Compose(
    [
        transforms.Resize(size=(input_width, input_height)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

encoder = CenternetEncoder(input_height, input_width)

dataset_val = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset_val)
torch_dataset = Dataset(dataset=dataset_val, transformation=transform, encoder=encoder)

training_data = torch_dataset
lr = 0.001
desired_loss = 0.1

if overfit:
    # randomly chosen pictures for model training
    # trainingdata_indices = torch.tensor([955, 1025, 219, 66, 1344, 222, 865, 2317, 86, 1409])

    trainingdata_indices = torch.randperm(len(torch_dataset))[:10]
    training_data = torch.utils.data.Subset(torch_dataset, trainingdata_indices)
    lr = 0.01
    desired_loss = 1.0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModelBuilder(alpha=0.25).to(device)

parameters = list(model.parameters())
optimizer = torch.optim.Adam(parameters, lr=lr)

model.train(True)

batch_generator = torch.utils.data.DataLoader(
    training_data, num_workers=4, batch_size=12
)

epoch = 1
get_desired_loss = False

while True:
    print("EPOCH {}:".format(epoch))
    for _, data in enumerate(batch_generator):
        input_data, gt_data = data
        input_data = input_data.to(device).contiguous()

        gt_data = gt_data.to(device)
        gt_data.requires_grad = False

        loss_dict = model(input_data, gt=gt_data)
        optimizer.zero_grad()  # compute gradient and do optimize step
        loss_dict["loss"].backward()

        optimizer.step()

        print(loss_dict["loss"])
        if loss_dict["loss"] < desired_loss:
            get_desired_loss = True
            break

    if get_desired_loss:
        break

    epoch += 1

torch.save(model.state_dict(), "../models/checkpoints/pretrained_weights.pt")

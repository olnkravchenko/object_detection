import torchvision
from torch.utils import data
import torchvision.transforms.v2 as transforms
import torch
from training.encoder import CenternetEncoder
from models.centernet import input_height, input_width, ModelBuilder
from data.dataset import Dataset

dataset_val = torchvision.datasets.VOCDetection(root="../VOC", year='2007', image_set="val", download=False)
dataset_val = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset_val)

trainingdata_indices = torch.tensor([955, 1025, 219, 66, 1344, 222, 865, 2317, 86, 1409])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ModelBuilder(alpha=0.25).to(device)

lr = 0.02
parameters = list(model.parameters())
optimizer = torch.optim.Adam(parameters, lr=lr)
encoder = CenternetEncoder(input_height, input_width)

EPOCHS = 5
model.train(True)

transform = transforms.Compose([
    transforms.Resize(size=(input_width, input_height)),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
])
torch_dataset = Dataset(dataset=dataset_val, transformation=transform, encoder=encoder)


training_data = torch.utils.data.Subset(torch_dataset, trainingdata_indices)

batch_generator = torch.utils.data.DataLoader(
    training_data,
    num_workers=4,
    batch_size=12
)

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))
    for _, data in enumerate(batch_generator):
        input_data, gt_data = data
        input_data = input_data.to(device).contiguous()

        gt_data = gt_data.to(device)
        gt_data.requires_grad = False

        loss_dict = model(input_data, gt=gt_data)
        optimizer.zero_grad() # compute gradient and do optimize step
        loss_dict['loss'].backward()

        optimizer.step()

        print(loss_dict['loss'])

# todo: we should define what we do with the result.
torch.save(model.state_dict(), "/path/to/training/model/result")

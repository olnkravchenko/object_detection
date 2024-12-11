import argparse
import os
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import multiprocessing
from data.dataset import Dataset
from models.centernet import ModelBuilder, input_height, input_width
from training.encoder import CenternetEncoder


def load_model_weights(model, checkpoint_path, continue_training=True):
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

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--overfit", action="store_true", help="overfit to 10 images")
args = parser.parse_args()

overfit = args.overfit

dataset_val = torchvision.datasets.VOCDetection(
    root="VOC", year="2007", image_set="val", download=False
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
lr = 0.03
batch_size = 32


def criteria_satisfied(_, current_epoch):
    if current_epoch >= 500:
        return True
    return False


if overfit:
    training_data = torch.utils.data.Subset(torch_dataset, range(10))
    lr = 1e-3
    batch_size = 14

    def criteria_satisfied(current_loss, _):
        if current_loss < 2000.0:
            return True
        return False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModelBuilder(alpha=0.25).to(device)

parameters = list(model.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=7,
    threshold=1e-4,
    threshold_mode="rel",
    cooldown=1,
    min_lr=1e-3,
)

model.train(True)

CONTINUE_TRAINING = True  # Set to False to start a new training process
CHECKPOINT_PATH = 'model_checkpoints/pascal_voc_final_weights.pt'

model, start_epoch = load_model_weights(model, CHECKPOINT_PATH, CONTINUE_TRAINING)

batch_generator = torch.utils.data.DataLoader(
    training_data, num_workers=0, batch_size=batch_size, shuffle=False
)

epoch = 1
get_desired_loss = False

os.makedirs('model_checkpoints', exist_ok=True)

while True:
    print("EPOCH {}:".format(epoch))

    loss_dict = {}
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

    if criteria_satisfied(loss_dict["loss"], epoch):
        break

    scheduler.step(loss_dict["loss"])

    epoch += 1

torch.save(model.state_dict(), "model_checkpoints/pascal_voc_final_weights.pt")


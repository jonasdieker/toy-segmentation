"""
Setup data loader
Loss
train func
val func
hyper-params
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
from tqdm import tqdm

from dataset import CustomDataset
from model import UNet

# perform argmax over channels of output
# apply crossentropy


def train_val_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: CrossEntropyLoss,
    epoch: int,
    device: str,
    optimizer: Optimizer = None,
    train: bool = True,
):
    running_loss = 0
    dice_score = []
    pbar = tqdm(dataloader)
    for i, data in enumerate(pbar):
        pbar.set_description(f"[{'Train' if train else 'Val'} epoch {str(epoch+1).zfill(3)}]")
        images, gts = data
        images = images.to(device).double()
        gts = gts.to(device)

        # zero gradients
        if train:
            optimizer.zero_grad()

        # pass data through model
        preds = model(images)
        loss = loss_fn(preds, gts)

        # compute gradients and adjust weights
        if train:
            loss.backward()
            optimizer.step()

        # update progress bar
        running_loss += loss.item()
        if i % 5 == 4:
            last_loss = running_loss / 5
            pbar.set_postfix({"loss": last_loss})
            running_loss = 0
        dice_score.append(dice(preds, gts))
    print(f"Dice Score: {sum(dice_score)/len(dice_score)}")


def main():
    batch_size = 8
    max_epochs = 20
    lr = 0.0001
    betas = (0.9, 0.999)
    num_classes = 12

    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss = CrossEntropyLoss()
    model = UNet(3, num_classes).double()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
    train_dataset = CustomDataset(
        image_root="/home/jonas/Downloads/CamVid/train",
        mask_root="/home/jonas/Downloads/CamVid/train_labels",
    )
    val_dataset = CustomDataset(
        image_root="/home/jonas/Downloads/CamVid/val",
        mask_root="/home/jonas/Downloads/CamVid/val_labels",
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4)

    for epoch in range(max_epochs):
        train_val_epoch(model, train_dataloader, loss, epoch, device, optimizer, train=True)
        train_val_epoch(model, val_dataloader, loss, epoch, device, train=False)

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()

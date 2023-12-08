"""
Setup data loader
Loss
train func
val func
hyper-params
"""
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from dataset import CustomDataset
from model import UNet

# perform argmax over channels of output
# apply crossentropy


def train_val_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: CrossEntropyLoss,
    epoch: int,
    optimizer: Optimizer = None,
    train: bool = True,
):
    running_loss = 0
    pbar = tqdm(dataloader)
    for i, data in enumerate(pbar):
        pbar.set_description(f"[Epoch {str(epoch+1).zfill(3)}]: ")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images, gts = data
        images = images.to(device).double()
        gts.to(device)

        # zero gradients
        if train:
            optimizer.zero_grad()

        # pass data through model
        preds = model(images)
        loss = loss_fn(preds, gts)
        loss.backward()

        # adjust weights
        if train:
            optimizer.step()

        # update progress bar
        running_loss += loss.item()
        if i % 5 == 4:
            last_loss = running_loss / 5
            pbar.postfix(f"loss: {last_loss}")
            running_loss = 0


def main():
    batch_size = 8
    max_epochs = 20
    lr = 0.0001
    betas = (0.9, 0.999)
    num_classes = 12

    loss = CrossEntropyLoss()
    model = UNet(3, num_classes).double()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
    train_dataset = CustomDataset(
        image_root="/home/jonas/Downloads/CamVid/train",
        mask_root="/home/jonas/Downloads/CamVid/trainannot",
    )
    val_dataset = CustomDataset(
        image_root="/home/jonas/Documents/data/CamVid/val",
        mask_root="/home/jonas/Documents/data/CamVid/valannot",
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4)

    for epoch in range(max_epochs):
        train_val_epoch(model, train_dataloader, loss, epoch, optimizer, train=True)
        train_val_epoch(model, val_dataloader, loss, epoch, train=False)

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
